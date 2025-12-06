
# api_processed_articles.py
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from supabase import create_client  # supabase-py

# ---------- Config from ENV ----------
SUPABASE_URL = 'https://jfdihmlxzemvdytrdcpw.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmZGlobWx4emVtdmR5dHJkY3B3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzgwNzE1NywiZXhwIjoyMDc5MzgzMTU3fQ.WDhd11X1G41ia7SclfnTg_DiEjpv6sGhZ071R7yGZKA'

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY environment variables before starting the API.")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Y2AI Processed Articles API (Supabase direct)")

# ---------- Response model ----------
class ArticleOut(BaseModel):
    id: int
    article_hash: Optional[str]
    source_type: Optional[str]
    source_name: Optional[str]
    title: Optional[str]
    url: Optional[str]
    published_at: Optional[str]
    processed_at: Optional[str]
    y2ai_category: Optional[str]
    impact_score: Optional[float]
    sentiment: Optional[str]
    keywords_used: Optional[List[str]]
    # you can add other fields if needed

class PaginatedResponse(BaseModel):
    total: Optional[int]
    limit: int
    offset: int
    data: List[ArticleOut]

# ---------- helpers ----------
def parse_iso(dt_str: str) -> datetime:
    """Parse common ISO formats and treat trailing Z as UTC"""
    try:
        if dt_str.endswith("Z"):
            dt_str = dt_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        # ensure tz-aware: if no tzinfo, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid ISO datetime: {dt_str}")

# ---------- endpoint ----------
@app.get("/processed_articles", response_model=PaginatedResponse)
def get_processed_articles(
    hours: Optional[int] = Query(None, ge=0, description="Fetch articles processed in last N hours"),
    processed_date: Optional[str] = Query(None, description="Either YYYY-MM-DD (date window, UTC) or ISO start time (e.g. 2025-12-06T16:00:00Z)"),
    impact_score: Optional[float] = Query(None, ge=0.0, description="Minimum impact score"),
    source_type: Optional[str] = Query(None),
    source_name: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None, description="Comma-separated sentiments, e.g. neutral,bullish"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Returns processed articles with optional filters:
     - hours=N  => processed_at >= now() - N hours
     - processed_date=YYYY-MM-DD => all articles with processed_at on that UTC date
     - processed_date=ISO_DATETIME => processed_at >= ISO_DATETIME
     - impact_score, source_type, source_name, sentiment
    """

    # Compute cutoff(s)
    cutoff_start_iso = None
    cutoff_end_iso = None  # used only when processed_date was YYYY-MM-DD

    if hours is not None:
        cutoff_dt = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=hours)
        cutoff_start_iso = cutoff_dt.isoformat()
    elif processed_date:
        # If user passed YYYY-MM-DD, treat as full-day UTC window:
        # start = YYYY-MM-DDT00:00:00+00:00
        # end   = next day YYYY-MM-DDT00:00:00+00:00 (exclusive)
        try:
            if len(processed_date) == 10:
                # assume date string YYYY-MM-DD
                date_obj = datetime.strptime(processed_date, "%Y-%m-%d").date()
                start_dt = datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=timezone.utc)
                next_start = start_dt + timedelta(days=1)
                cutoff_start_iso = start_dt.isoformat()
                cutoff_end_iso = next_start.isoformat()
            else:
                # treat as ISO datetime (start)
                start_dt = parse_iso(processed_date)
                cutoff_start_iso = start_dt.isoformat()
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid processed_date format. Use YYYY-MM-DD or ISO datetime.")

    try:
        # start building the supabase query
        query = sb.table("processed_articles").select("*", count="exact")

        # apply date filters
        if cutoff_start_iso and cutoff_end_iso:
            # full day window (>= start and < next_start)
            query = query.gte("processed_at", cutoff_start_iso)
            # use lt for exclusive upper bound; if .lt not available in client, consider .lte with minus microsecond
            query = query.lt("processed_at", cutoff_end_iso)
        elif cutoff_start_iso:
            query = query.gte("processed_at", cutoff_start_iso)

        if impact_score is not None:
            query = query.gte("impact_score", impact_score)

        if source_type:
            query = query.eq("source_type", source_type)

        if source_name:
            query = query.eq("source_name", source_name)

        if sentiment:
            sentiments = [s.strip() for s in sentiment.split(",") if s.strip()]
            if len(sentiments) == 1:
                query = query.eq("sentiment", sentiments[0])
            else:
                query = query.in_("sentiment", sentiments)

        # order + paginate
        query = query.order("processed_at", desc=True).range(offset, offset + limit - 1)

        resp = query.execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    data = resp.data or []
    total = getattr(resp, "count", None)

    # normalize to pydantic model shape (ArticleOut)
    out_items = []
    for r in data:
        # supabase returns JSON fields as Python objects already; pydantic will validate
        try:
            out_items.append(ArticleOut(**r))
        except Exception as e:
            # Defensive: if a record has unexpected shape, skip it with logging in production.
            raise HTTPException(status_code=500, detail=f"Record validation error: {e}")

    return PaginatedResponse(total=total, limit=limit, offset=offset, data=out_items)

