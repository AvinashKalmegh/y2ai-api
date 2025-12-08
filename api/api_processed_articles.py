
# # api_processed_articles.py
# import os
# from datetime import datetime, timedelta, timezone
# from typing import Optional, List

# from fastapi import FastAPI, Query, HTTPException
# from pydantic import BaseModel
# from supabase import create_client  # supabase-py

# # ---------- Config from ENV ----------
# SUPABASE_URL = 'https://jfdihmlxzemvdytrdcpw.supabase.co'
# SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmZGlobWx4emVtdmR5dHJkY3B3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzgwNzE1NywiZXhwIjoyMDc5MzgzMTU3fQ.WDhd11X1G41ia7SclfnTg_DiEjpv6sGhZ071R7yGZKA'

# if not SUPABASE_URL or not SUPABASE_KEY:
#     raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY environment variables before starting the API.")

# sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# app = FastAPI(title="Y2AI Processed Articles API (Supabase direct)")

# # ---------- Response model ----------
# class ArticleOut(BaseModel):
#     id: int
#     article_hash: Optional[str]
#     source_type: Optional[str]
#     source_name: Optional[str]
#     title: Optional[str]
#     url: Optional[str]
#     published_at: Optional[str]
#     processed_at: Optional[str]
#     y2ai_category: Optional[str]
#     impact_score: Optional[float]
#     sentiment: Optional[str]
#     keywords_used: Optional[List[str]]
#     # you can add other fields if needed

# class PaginatedResponse(BaseModel):
#     total: Optional[int]
#     limit: int
#     offset: int
#     data: List[ArticleOut]

# # ---------- helpers ----------
# def parse_iso(dt_str: str) -> datetime:
#     """Parse common ISO formats and treat trailing Z as UTC"""
#     try:
#         if dt_str.endswith("Z"):
#             dt_str = dt_str.replace("Z", "+00:00")
#         dt = datetime.fromisoformat(dt_str)
#         # ensure tz-aware: if no tzinfo, assume UTC
#         if dt.tzinfo is None:
#             dt = dt.replace(tzinfo=timezone.utc)
#         return dt
#     except Exception:
#         raise HTTPException(status_code=400, detail=f"Invalid ISO datetime: {dt_str}")

# # ---------- endpoint ----------
# @app.get("/processed_articles", response_model=PaginatedResponse)
# def get_processed_articles(
#     hours: Optional[int] = Query(None, ge=0, description="Fetch articles processed in last N hours"),
#     processed_date: Optional[str] = Query(None, description="Either YYYY-MM-DD (date window, UTC) or ISO start time (e.g. 2025-12-06T16:00:00Z)"),
#     impact_score: Optional[float] = Query(None, ge=0.0, description="Minimum impact score"),
#     source_type: Optional[str] = Query(None),
#     source_name: Optional[str] = Query(None),
#     sentiment: Optional[str] = Query(None, description="Comma-separated sentiments, e.g. neutral,bullish"),
#     limit: int = Query(100, ge=1, le=1000),
#     offset: int = Query(0, ge=0),
# ):
#     """
#     Returns processed articles with optional filters:
#      - hours=N  => processed_at >= now() - N hours
#      - processed_date=YYYY-MM-DD => all articles with processed_at on that UTC date
#      - processed_date=ISO_DATETIME => processed_at >= ISO_DATETIME
#      - impact_score, source_type, source_name, sentiment
#     """

#     # Compute cutoff(s)
#     cutoff_start_iso = None
#     cutoff_end_iso = None  # used only when processed_date was YYYY-MM-DD

#     if hours is not None:
#         cutoff_dt = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=hours)
#         cutoff_start_iso = cutoff_dt.isoformat()
#     elif processed_date:
#         # If user passed YYYY-MM-DD, treat as full-day UTC window:
#         # start = YYYY-MM-DDT00:00:00+00:00
#         # end   = next day YYYY-MM-DDT00:00:00+00:00 (exclusive)
#         try:
#             if len(processed_date) == 10:
#                 # assume date string YYYY-MM-DD
#                 date_obj = datetime.strptime(processed_date, "%Y-%m-%d").date()
#                 start_dt = datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=timezone.utc)
#                 next_start = start_dt + timedelta(days=1)
#                 cutoff_start_iso = start_dt.isoformat()
#                 cutoff_end_iso = next_start.isoformat()
#             else:
#                 # treat as ISO datetime (start)
#                 start_dt = parse_iso(processed_date)
#                 cutoff_start_iso = start_dt.isoformat()
#         except HTTPException:
#             raise
#         except Exception:
#             raise HTTPException(status_code=400, detail="Invalid processed_date format. Use YYYY-MM-DD or ISO datetime.")

#     try:
#         # start building the supabase query
#         query = sb.table("processed_articles").select("*", count="exact")

#         # apply date filters
#         if cutoff_start_iso and cutoff_end_iso:
#             # full day window (>= start and < next_start)
#             query = query.gte("published_at", cutoff_start_iso)
#             # use lt for exclusive upper bound; if .lt not available in client, consider .lte with minus microsecond
#             query = query.lt("published_at", cutoff_end_iso)
#         elif cutoff_start_iso:
#             query = query.gte("published_at", cutoff_start_iso)

#         if impact_score is not None:
#             query = query.gte("impact_score", impact_score)

#         if source_type:
#             query = query.eq("source_type", source_type)

#         if source_name:
#             query = query.eq("source_name", source_name)

#         if sentiment:
#             sentiments = [s.strip() for s in sentiment.split(",") if s.strip()]
#             if len(sentiments) == 1:
#                 query = query.eq("sentiment", sentiments[0])
#             else:
#                 query = query.in_("sentiment", sentiments)

#         # order + paginate
#         query = query.order("published_at", desc=True).range(offset, offset + limit - 1)

#         resp = query.execute()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"DB error: {e}")

#     data = resp.data or []
#     total = getattr(resp, "count", None)

#     # normalize to pydantic model shape (ArticleOut)
#     out_items = []
#     for r in data:
#         # supabase returns JSON fields as Python objects already; pydantic will validate
#         try:
#             out_items.append(ArticleOut(**r))
#         except Exception as e:
#             # Defensive: if a record has unexpected shape, skip it with logging in production.
#             raise HTTPException(status_code=500, detail=f"Record validation error: {e}")

#     return PaginatedResponse(total=total, limit=limit, offset=offset, data=out_items)



# api_processed_articles.py
"""
Y2AI Processed Articles API
Returns articles with nested signal structure for clean frontend consumption
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Any

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from supabase import create_client

# ---------- Config from ENV ----------
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://jfdihmlxzemvdytrdcpw.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmZGlobWx4emVtdmR5dHJkY3B3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzgwNzE1NywiZXhwIjoyMDc5MzgzMTU3fQ.WDhd11X1G41ia7SclfnTg_DiEjpv6sGhZ071R7yGZKA')

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY environment variables")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Y2AI Processed Articles API", version="2.0")


# =============================================================================
# NESTED RESPONSE MODELS
# =============================================================================

class ClassificationOut(BaseModel):
    y2ai_category: Optional[str] = None
    impact_score: Optional[float] = None
    sentiment: Optional[str] = None


class EntitiesOut(BaseModel):
    companies_mentioned: Optional[List[str]] = None
    tickers_mentioned: Optional[List[str]] = None
    dollar_amounts: Optional[List[str]] = None
    key_quotes: Optional[List[str]] = None
    extracted_facts: Optional[List[str]] = None


class CapexSignal(BaseModel):
    detected: bool = False
    direction: Optional[str] = None
    magnitude: Optional[str] = None
    company: Optional[str] = None
    amount: Optional[str] = None
    context: Optional[str] = None


class EnergySignal(BaseModel):
    detected: bool = False
    event_type: Optional[str] = None
    direction: Optional[str] = None
    region: Optional[str] = None
    context: Optional[str] = None


class ComputeSignal(BaseModel):
    detected: bool = False
    event_type: Optional[str] = None
    direction: Optional[str] = None
    companies_affected: Optional[List[str]] = None
    context: Optional[str] = None


class DepreciationSignal(BaseModel):
    detected: bool = False
    event_type: Optional[str] = None
    amount: Optional[str] = None
    company: Optional[str] = None
    context: Optional[str] = None


class VetoSignal(BaseModel):
    detected: bool = False
    trigger_type: Optional[str] = None
    severity: Optional[str] = None
    context: Optional[str] = None


class SignalsOut(BaseModel):
    capex: CapexSignal = CapexSignal()
    energy: EnergySignal = EnergySignal()
    compute: ComputeSignal = ComputeSignal()
    depreciation: DepreciationSignal = DepreciationSignal()
    veto: VetoSignal = VetoSignal()


class NewsletterOut(BaseModel):
    include_in_weekly: bool = False
    suggested_pillar: Optional[str] = None
    one_line_summary: Optional[str] = None


class ArticleOut(BaseModel):
    """Nested article structure for API response"""
    id: int
    article_hash: Optional[str] = None
    source_type: Optional[str] = None
    source_name: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[str] = None
    processed_at: Optional[str] = None
    keywords_used: Optional[List[str]] = None
    
    classification: ClassificationOut = ClassificationOut()
    entities: EntitiesOut = EntitiesOut()
    signals: SignalsOut = SignalsOut()
    newsletter: NewsletterOut = NewsletterOut()


class PaginatedResponse(BaseModel):
    total: Optional[int] = None
    limit: int
    offset: int
    data: List[ArticleOut]


class SignalSummary(BaseModel):
    """Aggregate signal counts for a time period"""
    period_start: str
    period_end: str
    total_articles: int
    capex_signals: int
    energy_signals: int
    compute_signals: int
    depreciation_signals: int
    veto_signals: int
    newsletter_worthy: int
    by_category: dict
    by_sentiment: dict
    by_pillar: dict


# =============================================================================
# HELPERS
# =============================================================================

def parse_iso(dt_str: str) -> datetime:
    """Parse common ISO formats and treat trailing Z as UTC"""
    try:
        if dt_str.endswith("Z"):
            dt_str = dt_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid ISO datetime: {dt_str}")


def transform_to_nested(row: dict) -> ArticleOut:
    """Transform flat DB row to nested API response structure"""
    
    # Helper to safely parse JSON fields that might be strings
    def safe_list(val) -> Optional[List[str]]:
        if val is None:
            return None
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                import json
                parsed = json.loads(val)
                return parsed if isinstance(parsed, list) else None
            except:
                return None
        return None
    
    # Helper to convert various bool representations
    def safe_bool(val) -> bool:
        if val is None:
            return False
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ('true', '1', 'yes')
        return bool(val)
    
    return ArticleOut(
        id=row.get("id"),
        article_hash=row.get("article_hash"),
        source_type=row.get("source_type"),
        source_name=row.get("source_name"),
        title=row.get("title"),
        url=row.get("url"),
        published_at=row.get("published_at"),
        processed_at=row.get("processed_at"),
        keywords_used=safe_list(row.get("keywords_used")),
        
        classification=ClassificationOut(
            y2ai_category=row.get("y2ai_category"),
            impact_score=row.get("impact_score"),
            sentiment=row.get("sentiment"),
        ),
        
        entities=EntitiesOut(
            companies_mentioned=safe_list(row.get("companies_mentioned")),
            tickers_mentioned=safe_list(row.get("tickers_mentioned")),
            dollar_amounts=safe_list(row.get("dollar_amounts")),
            key_quotes=safe_list(row.get("key_quotes")),
            extracted_facts=safe_list(row.get("extracted_facts")),
        ),
        
        signals=SignalsOut(
            capex=CapexSignal(
                detected=safe_bool(row.get("capex_detected")),
                direction=row.get("capex_direction"),
                magnitude=row.get("capex_magnitude"),
                company=row.get("capex_company"),
                amount=row.get("capex_amount"),
                context=row.get("capex_context"),
            ),
            energy=EnergySignal(
                detected=safe_bool(row.get("energy_detected")),
                event_type=row.get("energy_event_type"),
                direction=row.get("energy_direction"),
                region=row.get("energy_region"),
                context=row.get("energy_context"),
            ),
            compute=ComputeSignal(
                detected=safe_bool(row.get("compute_detected")),
                event_type=row.get("compute_event_type"),
                direction=row.get("compute_direction"),
                companies_affected=safe_list(row.get("compute_companies_affected")),
                context=row.get("compute_context"),
            ),
            depreciation=DepreciationSignal(
                detected=safe_bool(row.get("depreciation_detected")),
                event_type=row.get("depreciation_event_type"),
                amount=row.get("depreciation_amount"),
                company=row.get("depreciation_company"),
                context=row.get("depreciation_context"),
            ),
            veto=VetoSignal(
                detected=safe_bool(row.get("veto_detected")),
                trigger_type=row.get("veto_trigger_type"),
                severity=row.get("veto_severity"),
                context=row.get("veto_context"),
            ),
        ),
        
        newsletter=NewsletterOut(
            include_in_weekly=safe_bool(row.get("include_in_weekly")),
            suggested_pillar=row.get("suggested_pillar"),
            one_line_summary=row.get("one_line_summary"),
        ),
    )


# =============================================================================
# MAIN ENDPOINT
# =============================================================================

@app.get("/processed_articles", response_model=PaginatedResponse)
def get_processed_articles(
    # Time filters
    hours: Optional[int] = Query(None, ge=0, description="Articles from last N hours"),
    after: Optional[str] = Query(None, description="Articles after this date (YYYY-MM-DD or ISO datetime)"),
    before: Optional[str] = Query(None, description="Articles before this date (YYYY-MM-DD or ISO datetime)"),
    
    # Classification filters
    category: Optional[str] = Query(None, description="Y2AI category (spending, constraints, energy, etc.)"),
    impact_score_min: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum impact score"),
    sentiment: Optional[str] = Query(None, description="Comma-separated: bullish,bearish,neutral"),
    
    # Source filters
    source_type: Optional[str] = Query(None, description="Source type: rss, newsapi, etc."),
    source_name: Optional[str] = Query(None, description="Specific source name"),
    
    # Signal filters
    signal_type: Optional[str] = Query(None, description="Filter by signal: capex, energy, compute, depreciation, veto"),
    signal_detected: Optional[bool] = Query(None, description="True to get only detected signals"),
    
    # Newsletter filters
    include_in_weekly: Optional[bool] = Query(None, description="Filter newsletter-worthy articles"),
    suggested_pillar: Optional[str] = Query(None, description="Filter by pillar: software_margin, physical_assets, smart_money"),
    
    # Pagination
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Returns processed articles with nested signal structure.
    
    Signal filtering examples:
    - /processed_articles?signal_type=capex&signal_detected=true
    - /processed_articles?include_in_weekly=true&limit=50
    - /processed_articles?impact_score_min=0.7&category=spending
    """
    
    # Build date filters
    cutoff_start = None
    cutoff_end = None
    
    if hours is not None:
        cutoff_start = (datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=hours)).isoformat()
    
    if after:
        if len(after) == 10:  # YYYY-MM-DD
            date_obj = datetime.strptime(after, "%Y-%m-%d").date()
            cutoff_start = datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=timezone.utc).isoformat()
        else:
            cutoff_start = parse_iso(after).isoformat()
    
    if before:
        if len(before) == 10:  # YYYY-MM-DD
            date_obj = datetime.strptime(before, "%Y-%m-%d").date()
            cutoff_end = datetime.combine(date_obj, datetime.max.time()).replace(tzinfo=timezone.utc).isoformat()
        else:
            cutoff_end = parse_iso(before).isoformat()
    
    try:
        query = sb.table("processed_articles").select("*", count="exact")
        
        # Date filters
        if cutoff_start:
            query = query.gte("published_at", cutoff_start)
        if cutoff_end:
            query = query.lte("published_at", cutoff_end)
        
        # Classification filters
        if category:
            query = query.eq("y2ai_category", category)
        if impact_score_min is not None:
            query = query.gte("impact_score", impact_score_min)
        if sentiment:
            sentiments = [s.strip() for s in sentiment.split(",") if s.strip()]
            if len(sentiments) == 1:
                query = query.eq("sentiment", sentiments[0])
            else:
                query = query.in_("sentiment", sentiments)
        
        # Source filters
        if source_type:
            query = query.eq("source_type", source_type)
        if source_name:
            query = query.eq("source_name", source_name)
        
        # Signal filters
        if signal_type and signal_detected is not None:
            signal_column = f"{signal_type}_detected"
            query = query.eq(signal_column, signal_detected)
        elif signal_type and signal_detected is None:
            # If signal_type specified but signal_detected not, assume they want detected=true
            signal_column = f"{signal_type}_detected"
            query = query.eq(signal_column, True)
        
        # Newsletter filters
        if include_in_weekly is not None:
            query = query.eq("include_in_weekly", include_in_weekly)
        if suggested_pillar:
            query = query.eq("suggested_pillar", suggested_pillar)
        
        # Order and paginate
        query = query.order("published_at", desc=True).range(offset, offset + limit - 1)
        
        resp = query.execute()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
    data = resp.data or []
    total = getattr(resp, "count", None)
    
    # Transform flat rows to nested structure
    out_items = []
    for row in data:
        try:
            out_items.append(transform_to_nested(row))
        except Exception as e:
            # Log and skip malformed records in production
            print(f"Transform error for article {row.get('id')}: {e}")
            continue
    
    return PaginatedResponse(total=total, limit=limit, offset=offset, data=out_items)


# =============================================================================
# SIGNAL SUMMARY ENDPOINT
# =============================================================================

@app.get("/signals/summary", response_model=SignalSummary)
def get_signal_summary(
    after: Optional[str] = Query(None, description="Start date (YYYY-MM-DD or ISO)"),
    before: Optional[str] = Query(None, description="End date (YYYY-MM-DD or ISO)"),
    hours: Optional[int] = Query(168, ge=1, description="Default: last 168 hours (7 days)"),
):
    """
    Returns aggregate signal counts for a time period.
    Useful for dashboards and weekly summaries.
    """
    
    # Build date range
    if after and before:
        if len(after) == 10:
            start_dt = datetime.strptime(after, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            start_dt = parse_iso(after)
        if len(before) == 10:
            end_dt = datetime.strptime(before, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
        else:
            end_dt = parse_iso(before)
    else:
        end_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(hours=hours)
    
    try:
        resp = sb.table("processed_articles")\
            .select("*")\
            .gte("published_at", start_dt.isoformat())\
            .lte("published_at", end_dt.isoformat())\
            .execute()
        
        data = resp.data or []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
    # Compute aggregates
    capex_count = sum(1 for r in data if r.get("capex_detected"))
    energy_count = sum(1 for r in data if r.get("energy_detected"))
    compute_count = sum(1 for r in data if r.get("compute_detected"))
    depreciation_count = sum(1 for r in data if r.get("depreciation_detected"))
    veto_count = sum(1 for r in data if r.get("veto_detected"))
    newsletter_count = sum(1 for r in data if r.get("include_in_weekly"))
    
    # Category breakdown
    by_category = {}
    for r in data:
        cat = r.get("y2ai_category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    
    # Sentiment breakdown
    by_sentiment = {}
    for r in data:
        sent = r.get("sentiment", "unknown")
        by_sentiment[sent] = by_sentiment.get(sent, 0) + 1
    
    # Pillar breakdown (only for newsletter-worthy)
    by_pillar = {}
    for r in data:
        if r.get("include_in_weekly"):
            pillar = r.get("suggested_pillar") or "unassigned"
            by_pillar[pillar] = by_pillar.get(pillar, 0) + 1
    
    return SignalSummary(
        period_start=start_dt.isoformat(),
        period_end=end_dt.isoformat(),
        total_articles=len(data),
        capex_signals=capex_count,
        energy_signals=energy_count,
        compute_signals=compute_count,
        depreciation_signals=depreciation_count,
        veto_signals=veto_count,
        newsletter_worthy=newsletter_count,
        by_category=by_category,
        by_sentiment=by_sentiment,
        by_pillar=by_pillar,
    )


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
def health_check():
    """Basic health check endpoint"""
    try:
        # Quick DB ping
        sb.table("processed_articles").select("id").limit(1).execute()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e)}


# =============================================================================
# RUN WITH UVICORN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)