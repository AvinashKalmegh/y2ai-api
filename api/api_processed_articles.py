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
from fastapi.middleware.cors import CORSMiddleware


# ---------- Config from ENV ----------
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://jfdihmlxzemvdytrdcpw.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmZGlobWx4emVtdmR5dHJkY3B3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzgwNzE1NywiZXhwIjoyMDc5MzgzMTU3fQ.WDhd11X1G41ia7SclfnTg_DiEjpv6sGhZ071R7yGZKA')

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY environment variables")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Y2AI Processed Articles API", version="2.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",'https://y2ai-frontend.vercel.app'],  # Your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    processed_at: Optional[str] = None
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
    # New: specific date + time windows
    date: Optional[str] = Query(None, description="Specific date (YYYY-MM-DD) to filter by"),
    time_window: Optional[List[str]] = Query(None, description="Repeatable. Time window(s) in HH:MM-HH:MM format, e.g. 08:00-11:00"),
    
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
    Supports filtering by specific date and repeatable time windows (HH:MM-HH:MM).
    Example:
      /processed_articles?date=2025-12-11&time_window=08:00-11:00&time_window=15:00-18:00
    """
    # --- build base date range filters (match DB timestamp WITHOUT timezone format) ---
    cutoff_start = None
    cutoff_end = None

    if date:
        # user wants a specific date; use entire day range in DB query
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        cutoff_start = datetime.combine(date_obj, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
        cutoff_end = datetime.combine(date_obj, datetime.max.time()).strftime("%Y-%m-%d %H:%M:%S")
    else:
        # previous behavior: hours/after/before
        if hours is not None:
            cutoff_start = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

        if after:
            if len(after) == 10:  # YYYY-MM-DD
                date_obj = datetime.strptime(after, "%Y-%m-%d").date()
                cutoff_start = datetime.combine(date_obj, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
            else:
                parsed = parse_iso(after).replace(tzinfo=None)
                cutoff_start = parsed.strftime("%Y-%m-%d %H:%M:%S")

        if before:
            if len(before) == 10:
                date_obj = datetime.strptime(before, "%Y-%m-%d").date()
                cutoff_end = datetime.combine(date_obj, datetime.max.time()).strftime("%Y-%m-%d %H:%M:%S")
            else:
                parsed = parse_iso(before).replace(tzinfo=None)
                cutoff_end = parsed.strftime("%Y-%m-%d %H:%M:%S")

    # --- Parse time windows if provided (validate) ---
    # --- Parse time windows safely (normalize all dash types) ---
    parsed_windows = []
    if time_window:
        import re

        # normalize: convert en-dash, em-dash, minus, etc. → normal hyphen
        def normalize_dashes(s: str) -> str:
            return (
                s.replace('\u2013', '-')  # en-dash –
                .replace('\u2014', '-')  # em-dash —
                .replace('\u2212', '-')  # minus sign −
                .replace('\u2012', '-')  # figure dash
                .replace('\u2010', '-')  # hyphen
                .strip()
            )

        tw_re = re.compile(r'^([0-1]\d|2[0-3]):([0-5]\d)-([0-1]\d|2[0-3]):([0-5]\d)$')

        for win in time_window:
            win_norm = normalize_dashes(win)

            if not tw_re.match(win_norm):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid time_window format: {win}. Use HH:MM-HH:MM"
                )

            start_hm, end_hm = win_norm.split("-")
            start_dt = datetime.strptime(start_hm, "%H:%M").time()
            end_dt = datetime.strptime(end_hm, "%H:%M").time()
            parsed_windows.append((start_dt, end_dt))

    # --- Build Supabase query with DB-level date range only (if any) to reduce transferred rows ---
    try:
        query = sb.table("processed_articles").select("*", count="exact")
        if cutoff_start:
            query = query.gte("processed_at", cutoff_start)
        if cutoff_end:
            query = query.lte("processed_at", cutoff_end)

        # other filters (classification, sources, signals, newsletter) remain the same
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
        if source_type:
            query = query.eq("source_type", source_type)
        if source_name:
            query = query.eq("source_name", source_name)
        if signal_type and signal_detected is not None:
            signal_column = f"{signal_type}_detected"
            query = query.eq(signal_column, signal_detected)
        elif signal_type and signal_detected is None:
            signal_column = f"{signal_type}_detected"
            query = query.eq(signal_column, True)
        if include_in_weekly is not None:
            query = query.eq("include_in_weekly", include_in_weekly)
        if suggested_pillar:
            query = query.eq("suggested_pillar", suggested_pillar)

        # Do not range() here — we need to fetch all matching rows for in-app time-window filtering (page afterwards)
        resp = query.order("processed_at", desc=True).execute()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    rows = resp.data or []

    # --- If user requested time windows, filter rows in Python by processed_at time-of-day ---
    def parse_processed_at_val(v):
        # handle str or datetime
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        try:
            # accepted formats: 'YYYY-MM-DD HH:MM:SS[.ffffff]' or ISO with T
            s = str(v)
            # allow both space and T separators
            s = s.replace("T", " ")
            # Python's fromisoformat handles microseconds; try it first
            try:
                return datetime.fromisoformat(s)
            except Exception:
                # fallback parse with microseconds optional
                fmt_try = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
                for fmt in fmt_try:
                    try:
                        return datetime.strptime(s, fmt)
                    except:
                        continue
                raise
        except Exception:
            return None

    filtered_rows = []
    if parsed_windows:
        for r in rows:
            p = parse_processed_at_val(r.get("processed_at"))
            if p is None:
                continue
            p_time = p.time()
            # check any window matches (inclusive start, exclusive end)
            for (start_t, end_t) in parsed_windows:
                if start_t <= end_t:
                    if start_t <= p_time < end_t:
                        filtered_rows.append(r)
                        break
                else:
                    # window wraps midnight e.g., 23:00-02:00
                    if p_time >= start_t or p_time < end_t:
                        filtered_rows.append(r)
                        break
    else:
        filtered_rows = rows

    # --- Pagination applied after filtering ---
    total = len(filtered_rows)
    sliced = filtered_rows[offset: offset + limit]

    # Transform to nested
    out_items = []
    for row in sliced:
        try:
            out_items.append(transform_to_nested(row))
        except Exception as e:
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
            .gte("processed_at", start_dt.isoformat())\
            .lte("processed_at", end_dt.isoformat())\
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