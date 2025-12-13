# # api_processed_articles.py
# """
# Y2AI Processed Articles API
# Returns articles with nested signal structure for clean frontend consumption
# """

# import os
# from datetime import datetime, timedelta, timezone
# from typing import Optional, List, Any

# from fastapi import FastAPI, Query, HTTPException
# from pydantic import BaseModel
# from supabase import create_client
# from fastapi.middleware.cors import CORSMiddleware


# # ---------- Config from ENV ----------
# SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://jfdihmlxzemvdytrdcpw.supabase.co')
# SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmZGlobWx4emVtdmR5dHJkY3B3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzgwNzE1NywiZXhwIjoyMDc5MzgzMTU3fQ.WDhd11X1G41ia7SclfnTg_DiEjpv6sGhZ071R7yGZKA')

# if not SUPABASE_URL or not SUPABASE_KEY:
#     raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY environment variables")

# sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# app = FastAPI(title="Y2AI Processed Articles API", version="2.0")


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000",'https://y2ai-frontend.vercel.app'],  # Your React dev server
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # =============================================================================
# # NESTED RESPONSE MODELS
# # =============================================================================



# # ---- add these imports near the top of the file ----
# from pydantic import Field
# from typing import Union
# from datetime import date as date_class

# # ---- Pydantic model for daily signals ----
# class DailySignalsOut(BaseModel):
#     signal_date: str
#     capex_signal: Optional[float] = None
#     energy_signal: Optional[float] = None
#     compute_signal: Optional[float] = None
#     depreciation_signal: Optional[float] = None
#     thesis_balance: Optional[float] = None
#     veto_triggers: Optional[int] = None
#     signal_regime: Optional[str] = None
#     notes: Optional[str] = None
#     rows_aggregated: int = Field(..., description="Number of DB rows used to compute aggregates")
#     last_updated: Optional[str] = None

# # ---- helper to parse YYYY-MM-DD ----
# def parse_ymd(date_str: Optional[str]) -> date_class:
#     if date_str is None:
#         # default to today UTC date
#         return datetime.utcnow().date()
#     try:
#         return datetime.strptime(date_str, "%Y-%m-%d").date()
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

# # ---- daily signals endpoint (two routes point to same function) ----
# @app.get("/daily-signals", response_model=DailySignalsOut)
# def get_daily_signals(date: Optional[str] = Query(None, description="Date in YYYY-MM-DD (defaults to today UTC)")):
#     """
#     Fetch aggregated daily signals for a particular date.
#     Works with either /daily_signals or /daily-signals.
#     """
#     sig_date = parse_ymd(date)
#     day_start = datetime.combine(sig_date, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
#     day_end = datetime.combine(sig_date, datetime.max.time()).strftime("%Y-%m-%d %H:%M:%S")

#     # Column candidates to try in order (adjust these if your table uses different column names)
#     date_columns = ["date", "signal_date", "run_date"]

#     rows = []
#     last_err = None
#     try:
#         # First try direct equality queries on likely date columns
#         for col in date_columns:
#             try:
#                 resp = sb.table("daily_signals").select("*").eq(col, sig_date.isoformat()).execute()
#                 if resp.data:
#                     rows = resp.data
#                     break
#             except Exception as e:
#                 # ignore and try next candidate
#                 last_err = e
#                 continue

#         # If nothing found, fall back to matching within processed_at / created_at timestamp range
#         if not rows:
#             try:
#                 resp = sb.table("daily_signals")\
#                     .select("*")\
#                     .gte("processed_at", day_start)\
#                     .lte("processed_at", day_end)\
#                     .execute()
#                 rows = resp.data or []
#             except Exception as e:
#                 last_err = e
#                 rows = []

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Database error while fetching daily_signals: {e}")

#     if not rows:
#         # Return an empty/zero-style payload so frontend can show "no data" gracefully
#         return DailySignalsOut(
#             signal_date=sig_date.isoformat(),
#             capex_signal=None,
#             energy_signal=None,
#             compute_signal=None,
#             depreciation_signal=None,
#             thesis_balance=None,
#             veto_triggers=0,
#             signal_regime=None,
#             notes=f"No rows found for {sig_date.isoformat()}.",
#             rows_aggregated=0,
#             last_updated=None,
#         )

#     # numeric fields to average (continuous signals)
#     avg_fields = ["capex_signal", "energy_signal", "compute_signal", "depreciation_signal", "thesis_balance"]
#     # integer counters to sum
#     sum_fields = ["veto_triggers"]
#     # optional string fields - take most recent non-null value (by updated_at if present)
#     str_fields = ["signal_regime", "notes"]

#     # safe extractor
#     def safe_float(v):
#         try:
#             return float(v)
#         except Exception:
#             return None

#     def safe_int(v):
#         try:
#             return int(v)
#         except Exception:
#             return None

#     # compute averages
#     aggregates = {}
#     for f in avg_fields:
#         vals = [safe_float(r.get(f)) for r in rows if r.get(f) is not None]
#         vals = [v for v in vals if v is not None]
#         aggregates[f] = (sum(vals) / len(vals)) if vals else None

#     for f in sum_fields:
#         vals = [safe_int(r.get(f)) for r in rows if r.get(f) is not None]
#         vals = [v for v in vals if v is not None]
#         aggregates[f] = sum(vals) if vals else 0

#     # pick most-recent string by last_updated if available
#     for f in str_fields:
#         chosen = None
#         best_ts = None
#         for r in rows:
#             val = r.get(f)
#             if val:
#                 # prefer row with updated_at or created_at most recent
#                 ts = None
#                 for ts_col in ("updated_at", "last_updated", "created_at", "processed_at"):
#                     if r.get(ts_col):
#                         try:
#                             ts = datetime.fromisoformat(str(r.get(ts_col)).replace("Z", "+00:00"))
#                         except Exception:
#                             try:
#                                 ts = datetime.strptime(str(r.get(ts_col)), "%Y-%m-%d %H:%M:%S")
#                             except Exception:
#                                 ts = None
#                         if ts:
#                             break
#                 if best_ts is None or (ts is not None and ts > best_ts):
#                     best_ts = ts
#                     chosen = val
#         aggregates[f] = chosen

#     # determine last_updated across rows
#     last_updated = None
#     for r in rows:
#         for ts_col in ("updated_at", "last_updated", "processed_at", "created_at"):
#             if r.get(ts_col):
#                 try:
#                     cand = datetime.fromisoformat(str(r.get(ts_col)).replace("Z", "+00:00"))
#                 except Exception:
#                     try:
#                         cand = datetime.strptime(str(r.get(ts_col)), "%Y-%m-%d %H:%M:%S")
#                     except Exception:
#                         cand = None
#                 if cand and (last_updated is None or cand > last_updated):
#                     last_updated = cand

#     return DailySignalsOut(
#         signal_date=sig_date.isoformat(),
#         capex_signal=aggregates.get("capex_signal"),
#         energy_signal=aggregates.get("energy_signal"),
#         compute_signal=aggregates.get("compute_signal"),
#         depreciation_signal=aggregates.get("depreciation_signal"),
#         thesis_balance=aggregates.get("thesis_balance"),
#         veto_triggers=aggregates.get("veto_triggers"),
#         signal_regime=aggregates.get("signal_regime"),
#         notes=aggregates.get("notes"),
#         rows_aggregated=len(rows),
#         last_updated=last_updated.isoformat() if last_updated else None,
#     )


# class ClassificationOut(BaseModel):
#     y2ai_category: Optional[str] = None
#     impact_score: Optional[float] = None
#     sentiment: Optional[str] = None


# class EntitiesOut(BaseModel):
#     companies_mentioned: Optional[List[str]] = None
#     tickers_mentioned: Optional[List[str]] = None
#     dollar_amounts: Optional[List[str]] = None
#     key_quotes: Optional[List[str]] = None
#     extracted_facts: Optional[List[str]] = None


# class CapexSignal(BaseModel):
#     detected: bool = False
#     direction: Optional[str] = None
#     magnitude: Optional[str] = None
#     company: Optional[str] = None
#     amount: Optional[str] = None
#     context: Optional[str] = None


# class EnergySignal(BaseModel):
#     detected: bool = False
#     event_type: Optional[str] = None
#     direction: Optional[str] = None
#     region: Optional[str] = None
#     context: Optional[str] = None


# class ComputeSignal(BaseModel):
#     detected: bool = False
#     event_type: Optional[str] = None
#     direction: Optional[str] = None
#     companies_affected: Optional[List[str]] = None
#     context: Optional[str] = None


# class DepreciationSignal(BaseModel):
#     detected: bool = False
#     event_type: Optional[str] = None
#     amount: Optional[str] = None
#     company: Optional[str] = None
#     context: Optional[str] = None


# class VetoSignal(BaseModel):
#     detected: bool = False
#     trigger_type: Optional[str] = None
#     severity: Optional[str] = None
#     context: Optional[str] = None


# class SignalsOut(BaseModel):
#     capex: CapexSignal = CapexSignal()
#     energy: EnergySignal = EnergySignal()
#     compute: ComputeSignal = ComputeSignal()
#     depreciation: DepreciationSignal = DepreciationSignal()
#     veto: VetoSignal = VetoSignal()


# class NewsletterOut(BaseModel):
#     include_in_weekly: bool = False
#     suggested_pillar: Optional[str] = None
#     one_line_summary: Optional[str] = None


# class ArticleOut(BaseModel):
#     """Nested article structure for API response"""
#     id: int
#     article_hash: Optional[str] = None
#     source_type: Optional[str] = None
#     source_name: Optional[str] = None
#     title: Optional[str] = None
#     url: Optional[str] = None
#     processed_at: Optional[str] = None
#     processed_at: Optional[str] = None
#     keywords_used: Optional[List[str]] = None
    
#     classification: ClassificationOut = ClassificationOut()
#     entities: EntitiesOut = EntitiesOut()
#     signals: SignalsOut = SignalsOut()
#     newsletter: NewsletterOut = NewsletterOut()


# class PaginatedResponse(BaseModel):
#     total: Optional[int] = None
#     limit: int
#     offset: int
#     data: List[ArticleOut]


# class SignalSummary(BaseModel):
#     """Aggregate signal counts for a time period"""
#     period_start: str
#     period_end: str
#     total_articles: int
#     capex_signals: int
#     energy_signals: int
#     compute_signals: int
#     depreciation_signals: int
#     veto_signals: int
#     newsletter_worthy: int
#     by_category: dict
#     by_sentiment: dict
#     by_pillar: dict


# # =============================================================================
# # HELPERS
# # =============================================================================

# def parse_iso(dt_str: str) -> datetime:
#     """Parse common ISO formats and treat trailing Z as UTC"""
#     try:
#         if dt_str.endswith("Z"):
#             dt_str = dt_str.replace("Z", "+00:00")
#         dt = datetime.fromisoformat(dt_str)
#         if dt.tzinfo is None:
#             dt = dt.replace(tzinfo=timezone.utc)
#         return dt
#     except Exception:
#         raise HTTPException(status_code=400, detail=f"Invalid ISO datetime: {dt_str}")


# def transform_to_nested(row: dict) -> ArticleOut:
#     """Transform flat DB row to nested API response structure"""
    
#     # Helper to safely parse JSON fields that might be strings
#     def safe_list(val) -> Optional[List[str]]:
#         if val is None:
#             return None
#         if isinstance(val, list):
#             return val
#         if isinstance(val, str):
#             try:
#                 import json
#                 parsed = json.loads(val)
#                 return parsed if isinstance(parsed, list) else None
#             except:
#                 return None
#         return None
    
#     # Helper to convert various bool representations
#     def safe_bool(val) -> bool:
#         if val is None:
#             return False
#         if isinstance(val, bool):
#             return val
#         if isinstance(val, str):
#             return val.lower() in ('true', '1', 'yes')
#         return bool(val)
    
#     return ArticleOut(
#         id=row.get("id"),
#         article_hash=row.get("article_hash"),
#         source_type=row.get("source_type"),
#         source_name=row.get("source_name"),
#         title=row.get("title"),
#         url=row.get("url"),
#         published_at=row.get("published_at"),
#         processed_at=row.get("processed_at"),
#         keywords_used=safe_list(row.get("keywords_used")),
        
#         classification=ClassificationOut(
#             y2ai_category=row.get("y2ai_category"),
#             impact_score=row.get("impact_score"),
#             sentiment=row.get("sentiment"),
#         ),
        
#         entities=EntitiesOut(
#             companies_mentioned=safe_list(row.get("companies_mentioned")),
#             tickers_mentioned=safe_list(row.get("tickers_mentioned")),
#             dollar_amounts=safe_list(row.get("dollar_amounts")),
#             key_quotes=safe_list(row.get("key_quotes")),
#             extracted_facts=safe_list(row.get("extracted_facts")),
#         ),
        
#         signals=SignalsOut(
#             capex=CapexSignal(
#                 detected=safe_bool(row.get("capex_detected")),
#                 direction=row.get("capex_direction"),
#                 magnitude=row.get("capex_magnitude"),
#                 company=row.get("capex_company"),
#                 amount=row.get("capex_amount"),
#                 context=row.get("capex_context"),
#             ),
#             energy=EnergySignal(
#                 detected=safe_bool(row.get("energy_detected")),
#                 event_type=row.get("energy_event_type"),
#                 direction=row.get("energy_direction"),
#                 region=row.get("energy_region"),
#                 context=row.get("energy_context"),
#             ),
#             compute=ComputeSignal(
#                 detected=safe_bool(row.get("compute_detected")),
#                 event_type=row.get("compute_event_type"),
#                 direction=row.get("compute_direction"),
#                 companies_affected=safe_list(row.get("compute_companies_affected")),
#                 context=row.get("compute_context"),
#             ),
#             depreciation=DepreciationSignal(
#                 detected=safe_bool(row.get("depreciation_detected")),
#                 event_type=row.get("depreciation_event_type"),
#                 amount=row.get("depreciation_amount"),
#                 company=row.get("depreciation_company"),
#                 context=row.get("depreciation_context"),
#             ),
#             veto=VetoSignal(
#                 detected=safe_bool(row.get("veto_detected")),
#                 trigger_type=row.get("veto_trigger_type"),
#                 severity=row.get("veto_severity"),
#                 context=row.get("veto_context"),
#             ),
#         ),
        
#         newsletter=NewsletterOut(
#             include_in_weekly=safe_bool(row.get("include_in_weekly")),
#             suggested_pillar=row.get("suggested_pillar"),
#             one_line_summary=row.get("one_line_summary"),
#         ),
#     )


# # =============================================================================
# # MAIN ENDPOINT
# # =============================================================================

# @app.get("/processed_articles", response_model=PaginatedResponse)
# def get_processed_articles(
#     # Time filters
#     hours: Optional[int] = Query(None, ge=0, description="Articles from last N hours"),
#     after: Optional[str] = Query(None, description="Articles after this date (YYYY-MM-DD or ISO datetime)"),
#     before: Optional[str] = Query(None, description="Articles before this date (YYYY-MM-DD or ISO datetime)"),
#     # New: specific date + time windows
#     date: Optional[str] = Query(None, description="Specific date (YYYY-MM-DD) to filter by"),
#     time_window: Optional[List[str]] = Query(None, description="Repeatable. Time window(s) in HH:MM-HH:MM format, e.g. 08:00-11:00"),
    
#     # Classification filters
#     category: Optional[str] = Query(None, description="Y2AI category (spending, constraints, energy, etc.)"),
#     impact_score_min: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum impact score"),
#     sentiment: Optional[str] = Query(None, description="Comma-separated: bullish,bearish,neutral"),
    
#     # Source filters
#     source_type: Optional[str] = Query(None, description="Source type: rss, newsapi, etc."),
#     source_name: Optional[str] = Query(None, description="Specific source name"),
    
#     # Signal filters
#     signal_type: Optional[str] = Query(None, description="Filter by signal: capex, energy, compute, depreciation, veto"),
#     signal_detected: Optional[bool] = Query(None, description="True to get only detected signals"),
    
#     # Newsletter filters
#     include_in_weekly: Optional[bool] = Query(None, description="Filter newsletter-worthy articles"),
#     suggested_pillar: Optional[str] = Query(None, description="Filter by pillar: software_margin, physical_assets, smart_money"),
    
#     # Pagination
#     limit: int = Query(100, ge=1, le=1000),
#     offset: int = Query(0, ge=0),
# ):
#     """
#     Returns processed articles with nested signal structure.
#     Supports filtering by specific date and repeatable time windows (HH:MM-HH:MM).
#     Example:
#       /processed_articles?date=2025-12-11&time_window=08:00-11:00&time_window=15:00-18:00
#     """
#     # --- build base date range filters (match DB timestamp WITHOUT timezone format) ---
#     cutoff_start = None
#     cutoff_end = None

#     if date:
#         # user wants a specific date; use entire day range in DB query
#         try:
#             date_obj = datetime.strptime(date, "%Y-%m-%d").date()
#         except Exception:
#             raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
#         cutoff_start = datetime.combine(date_obj, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
#         cutoff_end = datetime.combine(date_obj, datetime.max.time()).strftime("%Y-%m-%d %H:%M:%S")
#     else:
#         # previous behavior: hours/after/before
#         if hours is not None:
#             cutoff_start = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

#         if after:
#             if len(after) == 10:  # YYYY-MM-DD
#                 date_obj = datetime.strptime(after, "%Y-%m-%d").date()
#                 cutoff_start = datetime.combine(date_obj, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
#             else:
#                 parsed = parse_iso(after).replace(tzinfo=None)
#                 cutoff_start = parsed.strftime("%Y-%m-%d %H:%M:%S")

#         if before:
#             if len(before) == 10:
#                 date_obj = datetime.strptime(before, "%Y-%m-%d").date()
#                 cutoff_end = datetime.combine(date_obj, datetime.max.time()).strftime("%Y-%m-%d %H:%M:%S")
#             else:
#                 parsed = parse_iso(before).replace(tzinfo=None)
#                 cutoff_end = parsed.strftime("%Y-%m-%d %H:%M:%S")

#     # --- Parse time windows if provided (validate) ---
#     # --- Parse time windows safely (normalize all dash types) ---
#     parsed_windows = []
#     if time_window:
#         import re

#         # normalize: convert en-dash, em-dash, minus, etc. → normal hyphen
#         def normalize_dashes(s: str) -> str:
#             return (
#                 s.replace('\u2013', '-')  # en-dash –
#                 .replace('\u2014', '-')  # em-dash —
#                 .replace('\u2212', '-')  # minus sign −
#                 .replace('\u2012', '-')  # figure dash
#                 .replace('\u2010', '-')  # hyphen
#                 .strip()
#             )

#         tw_re = re.compile(r'^([0-1]\d|2[0-3]):([0-5]\d)-([0-1]\d|2[0-3]):([0-5]\d)$')

#         for win in time_window:
#             win_norm = normalize_dashes(win)

#             if not tw_re.match(win_norm):
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Invalid time_window format: {win}. Use HH:MM-HH:MM"
#                 )

#             start_hm, end_hm = win_norm.split("-")
#             start_dt = datetime.strptime(start_hm, "%H:%M").time()
#             end_dt = datetime.strptime(end_hm, "%H:%M").time()
#             parsed_windows.append((start_dt, end_dt))

#     # --- Build Supabase query with DB-level date range only (if any) to reduce transferred rows ---
#     try:
#         query = sb.table("processed_articles").select("*", count="exact")
#         if cutoff_start:
#             query = query.gte("processed_at", cutoff_start)
#         if cutoff_end:
#             query = query.lte("processed_at", cutoff_end)

#         # other filters (classification, sources, signals, newsletter) remain the same
#         if category:
#             query = query.eq("y2ai_category", category)
#         if impact_score_min is not None:
#             query = query.gte("impact_score", impact_score_min)
#         if sentiment:
#             sentiments = [s.strip() for s in sentiment.split(",") if s.strip()]
#             if len(sentiments) == 1:
#                 query = query.eq("sentiment", sentiments[0])
#             else:
#                 query = query.in_("sentiment", sentiments)
#         if source_type:
#             query = query.eq("source_type", source_type)
#         if source_name:
#             query = query.eq("source_name", source_name)
#         if signal_type and signal_detected is not None:
#             signal_column = f"{signal_type}_detected"
#             query = query.eq(signal_column, signal_detected)
#         elif signal_type and signal_detected is None:
#             signal_column = f"{signal_type}_detected"
#             query = query.eq(signal_column, True)
#         if include_in_weekly is not None:
#             query = query.eq("include_in_weekly", include_in_weekly)
#         if suggested_pillar:
#             query = query.eq("suggested_pillar", suggested_pillar)

#         # Do not range() here — we need to fetch all matching rows for in-app time-window filtering (page afterwards)
#         resp = query.order("processed_at", desc=True).execute()

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")

#     rows = resp.data or []

#     # --- If user requested time windows, filter rows in Python by processed_at time-of-day ---
#     def parse_processed_at_val(v):
#         # handle str or datetime
#         if v is None:
#             return None
#         if isinstance(v, datetime):
#             return v
#         try:
#             # accepted formats: 'YYYY-MM-DD HH:MM:SS[.ffffff]' or ISO with T
#             s = str(v)
#             # allow both space and T separators
#             s = s.replace("T", " ")
#             # Python's fromisoformat handles microseconds; try it first
#             try:
#                 return datetime.fromisoformat(s)
#             except Exception:
#                 # fallback parse with microseconds optional
#                 fmt_try = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
#                 for fmt in fmt_try:
#                     try:
#                         return datetime.strptime(s, fmt)
#                     except:
#                         continue
#                 raise
#         except Exception:
#             return None

#     filtered_rows = []
#     if parsed_windows:
#         for r in rows:
#             p = parse_processed_at_val(r.get("processed_at"))
#             if p is None:
#                 continue
#             p_time = p.time()
#             # check any window matches (inclusive start, exclusive end)
#             for (start_t, end_t) in parsed_windows:
#                 if start_t <= end_t:
#                     if start_t <= p_time < end_t:
#                         filtered_rows.append(r)
#                         break
#                 else:
#                     # window wraps midnight e.g., 23:00-02:00
#                     if p_time >= start_t or p_time < end_t:
#                         filtered_rows.append(r)
#                         break
#     else:
#         filtered_rows = rows

#     # --- Pagination applied after filtering ---
#     total = len(filtered_rows)
#     sliced = filtered_rows[offset: offset + limit]

#     # Transform to nested
#     out_items = []
#     for row in sliced:
#         try:
#             out_items.append(transform_to_nested(row))
#         except Exception as e:
#             print(f"Transform error for article {row.get('id')}: {e}")
#             continue

#     return PaginatedResponse(total=total, limit=limit, offset=offset, data=out_items)


# # =============================================================================
# # SIGNAL SUMMARY ENDPOINT
# # =============================================================================

# @app.get("/signals/summary", response_model=SignalSummary)
# def get_signal_summary(
#     after: Optional[str] = Query(None, description="Start date (YYYY-MM-DD or ISO)"),
#     before: Optional[str] = Query(None, description="End date (YYYY-MM-DD or ISO)"),
#     hours: Optional[int] = Query(168, ge=1, description="Default: last 168 hours (7 days)"),
# ):
#     """
#     Returns aggregate signal counts for a time period.
#     Useful for dashboards and weekly summaries.
#     """
    
#     # Build date range
#     if after and before:
#         if len(after) == 10:
#             start_dt = datetime.strptime(after, "%Y-%m-%d").replace(tzinfo=timezone.utc)
#         else:
#             start_dt = parse_iso(after)
#         if len(before) == 10:
#             end_dt = datetime.strptime(before, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
#         else:
#             end_dt = parse_iso(before)
#     else:
#         end_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
#         start_dt = end_dt - timedelta(hours=hours)
    
#     try:
#         resp = sb.table("processed_articles")\
#             .select("*")\
#             .gte("processed_at", start_dt.isoformat())\
#             .lte("processed_at", end_dt.isoformat())\
#             .execute()
        
#         data = resp.data or []
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
#     # Compute aggregates
#     capex_count = sum(1 for r in data if r.get("capex_detected"))
#     energy_count = sum(1 for r in data if r.get("energy_detected"))
#     compute_count = sum(1 for r in data if r.get("compute_detected"))
#     depreciation_count = sum(1 for r in data if r.get("depreciation_detected"))
#     veto_count = sum(1 for r in data if r.get("veto_detected"))
#     newsletter_count = sum(1 for r in data if r.get("include_in_weekly"))
    
#     # Category breakdown
#     by_category = {}
#     for r in data:
#         cat = r.get("y2ai_category", "unknown")
#         by_category[cat] = by_category.get(cat, 0) + 1
    
#     # Sentiment breakdown
#     by_sentiment = {}
#     for r in data:
#         sent = r.get("sentiment", "unknown")
#         by_sentiment[sent] = by_sentiment.get(sent, 0) + 1
    
#     # Pillar breakdown (only for newsletter-worthy)
#     by_pillar = {}
#     for r in data:
#         if r.get("include_in_weekly"):
#             pillar = r.get("suggested_pillar") or "unassigned"
#             by_pillar[pillar] = by_pillar.get(pillar, 0) + 1
    
#     return SignalSummary(
#         period_start=start_dt.isoformat(),
#         period_end=end_dt.isoformat(),
#         total_articles=len(data),
#         capex_signals=capex_count,
#         energy_signals=energy_count,
#         compute_signals=compute_count,
#         depreciation_signals=depreciation_count,
#         veto_signals=veto_count,
#         newsletter_worthy=newsletter_count,
#         by_category=by_category,
#         by_sentiment=by_sentiment,
#         by_pillar=by_pillar,
#     )


# # =============================================================================
# # HEALTH CHECK
# # =============================================================================

# @app.get("/health")
# def health_check():
#     """Basic health check endpoint"""
#     try:
#         # Quick DB ping
#         sb.table("processed_articles").select("id").limit(1).execute()
#         return {"status": "healthy", "database": "connected"}
#     except Exception as e:
#         return {"status": "unhealthy", "database": str(e)}


# # =============================================================================
# # RUN WITH UVICORN
# # =============================================================================

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



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
from zoneinfo import ZoneInfo

DISPLAY_TZ = ZoneInfo("America/New_York")  # EST/EDT

def convert_est_to_utc_for_query(date_str: str, is_start: bool = True):
    """
    Convert EST/EDT date string to UTC for Supabase queries.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        is_start: True for day start (00:00:00), False for day end (23:59:59)
    
    Returns:
        UTC datetime string for Supabase query
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    if is_start:
        local_dt = datetime.combine(date_obj, datetime.min.time())
    else:
        local_dt = datetime.combine(date_obj, datetime.max.time())
    
    # Attach EST/EDT timezone
    local_dt = local_dt.replace(tzinfo=DISPLAY_TZ)
    
    # Convert to UTC
    utc_dt = local_dt.astimezone(timezone.utc)
    
    return utc_dt.isoformat()


def to_display_tz(dt_val):
    if not dt_val:
        return None

    if isinstance(dt_val, datetime):
        dt = dt_val
    else:
        s = str(dt_val).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(DISPLAY_TZ).isoformat()



# ---------- Config from ENV ----------
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://jfdihmlxzemvdytrdcpw.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmZGlobWx4emVtdmR5dHJkY3B3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzgwNzE1NywiZXhwIjoyMDc5MzgzMTU3fQ.WDhd11X1G41ia7SclfnTg_DiEjpv6sGhZ071R7yGZKA')

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY environment variables")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Y2AI Processed Articles API", version="2.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",'https://y2ai-frontend.vercel.app'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# METRICS HISTORY MODELS AND ENDPOINT
# =============================================================================

class MetricsHistoryRow(BaseModel):
    date: str
    amri: Optional[float] = None
    amri_regime: Optional[str] = None
    enhanced_amri: Optional[float] = None
    break_prob: Optional[float] = None
    mci: Optional[float] = None
    mci_regime: Optional[str] = None
    bubble_index: Optional[float] = None
    clusters: Optional[int] = None
    corr_20d: Optional[float] = None
    breadth_20d: Optional[float] = None
    vix: Optional[float] = None
    credit_spreads: Optional[float] = None
    infra_breadth: Optional[float] = None
    ent_breadth: Optional[float] = None
    thesis_balance: Optional[float] = None


class MetricsHistoryResponse(BaseModel):
    total: int
    data: List[MetricsHistoryRow]


@app.get("/metrics-history", response_model=MetricsHistoryResponse)
def get_metrics_history(
    days: int = Query(90, ge=1, le=365, description="Number of days of history to return"),
    after: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    before: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Returns historical metrics for charting (AMRI, Bubble Index, MCI, Breadth, etc.)
    Default: last 90 days, ordered by date ascending for chart rendering.
    """
    try:
        query = sb.table("metrics_history").select("*")
        
        # Date filtering
        if after:
            query = query.gte("date", after)
        if before:
            query = query.lte("date", before)
        
        # If no explicit date range, use days parameter
        if not after and not before:
            cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            query = query.gte("date", cutoff)
        
        # Order ascending for charts (oldest first)
        query = query.order("date", desc=False)
        
        resp = query.execute()
        rows = resp.data or []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
    # Transform to response model
    out = []
    for r in rows:
        out.append(MetricsHistoryRow(
            date=str(r.get("date", "")),
            amri=r.get("amri"),
            amri_regime=r.get("amri_regime"),
            enhanced_amri=r.get("enhanced_amri"),
            break_prob=r.get("break_prob"),
            mci=r.get("mci"),
            mci_regime=r.get("mci_regime"),
            bubble_index=r.get("bubble_index"),
            clusters=r.get("clusters"),
            corr_20d=r.get("corr_20d"),
            breadth_20d=r.get("breadth_20d"),
            vix=r.get("vix"),
            credit_spreads=r.get("credit_spreads"),
            infra_breadth=r.get("infra_breadth"),
            ent_breadth=r.get("ent_breadth"),
            thesis_balance=r.get("thesis_balance"),
        ))
    
    return MetricsHistoryResponse(total=len(out), data=out)


@app.post("/metrics-history")
def upsert_metrics_history(row: MetricsHistoryRow):
    """
    Insert or update a single day's metrics.
    Used by Google Apps Script or Python to push daily snapshots.
    """
    try:
        data = {
            "date": row.date,
            "amri": row.amri,
            "amri_regime": row.amri_regime,
            "enhanced_amri": row.enhanced_amri,
            "break_prob": row.break_prob,
            "mci": row.mci,
            "mci_regime": row.mci_regime,
            "bubble_index": row.bubble_index,
            "clusters": row.clusters,
            "corr_20d": row.corr_20d,
            "breadth_20d": row.breadth_20d,
            "vix": row.vix,
            "credit_spreads": row.credit_spreads,
            "infra_breadth": row.infra_breadth,
            "ent_breadth": row.ent_breadth,
            "thesis_balance": row.thesis_balance,
        }
        
        resp = sb.table("metrics_history").upsert(data, on_conflict="date").execute()
        return {"status": "ok", "date": row.date}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.post("/metrics-history/bulk")
def bulk_upsert_metrics_history(rows: List[MetricsHistoryRow]):
    """
    Bulk insert/update metrics history.
    Used for seeding historical data from Google Sheets export.
    """
    try:
        data = []
        for row in rows:
            data.append({
                "date": row.date,
                "amri": row.amri,
                "amri_regime": row.amri_regime,
                "enhanced_amri": row.enhanced_amri,
                "break_prob": row.break_prob,
                "mci": row.mci,
                "mci_regime": row.mci_regime,
                "bubble_index": row.bubble_index,
                "clusters": row.clusters,
                "corr_20d": row.corr_20d,
                "breadth_20d": row.breadth_20d,
                "vix": row.vix,
                "credit_spreads": row.credit_spreads,
                "infra_breadth": row.infra_breadth,
                "ent_breadth": row.ent_breadth,
                "thesis_balance": row.thesis_balance,
            })
        
        resp = sb.table("metrics_history").upsert(data, on_conflict="date").execute()
        return {"status": "ok", "rows_processed": len(data)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


# =============================================================================
# NESTED RESPONSE MODELS (existing code below)
# =============================================================================

from pydantic import Field
from typing import Union
from datetime import date as date_class

class DailySignalsOut(BaseModel):
    signal_date: str
    capex_signal: Optional[float] = None
    energy_signal: Optional[float] = None
    compute_signal: Optional[float] = None
    depreciation_signal: Optional[float] = None
    thesis_balance: Optional[float] = None
    veto_triggers: Optional[int] = None
    signal_regime: Optional[str] = None
    notes: Optional[str] = None
    rows_aggregated: int = Field(..., description="Number of DB rows used to compute aggregates")
    last_updated: Optional[str] = None


def parse_ymd(date_str: Optional[str]) -> date_class:
    if date_str is None:
        return datetime.utcnow().date()
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")


@app.get("/daily-signals", response_model=DailySignalsOut)
def get_daily_signals(date: Optional[str] = Query(None, description="Date in YYYY-MM-DD (defaults to today UTC)")):
    """
    Fetch aggregated daily signals for a particular date.
    """
    sig_date = parse_ymd(date)
    day_start = datetime.combine(sig_date, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
    day_end = datetime.combine(sig_date, datetime.max.time()).strftime("%Y-%m-%d %H:%M:%S")

    date_columns = ["date", "signal_date", "run_date"]

    rows = []
    last_err = None
    try:
        for col in date_columns:
            try:
                resp = sb.table("daily_signals").select("*").eq(col, sig_date.isoformat()).execute()
                if resp.data:
                    rows = resp.data
                    break
            except Exception as e:
                last_err = e
                continue

        if not rows:
            try:
                resp = sb.table("daily_signals")\
                    .select("*")\
                    .gte("processed_at", day_start)\
                    .lte("processed_at", day_end)\
                    .execute()
                rows = resp.data or []
            except Exception as e:
                last_err = e
                rows = []

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error while fetching daily_signals: {e}")

    if not rows:
        return DailySignalsOut(
            signal_date=sig_date.isoformat(),
            capex_signal=None,
            energy_signal=None,
            compute_signal=None,
            depreciation_signal=None,
            thesis_balance=None,
            veto_triggers=0,
            signal_regime=None,
            notes=f"No rows found for {sig_date.isoformat()}.",
            rows_aggregated=0,
            last_updated=None,
        )

    avg_fields = ["capex_signal", "energy_signal", "compute_signal", "depreciation_signal", "thesis_balance"]
    sum_fields = ["veto_triggers"]
    str_fields = ["signal_regime", "notes"]

    def safe_float(v):
        try:
            return float(v)
        except Exception:
            return None

    def safe_int(v):
        try:
            return int(v)
        except Exception:
            return None

    aggregates = {}
    for f in avg_fields:
        vals = [safe_float(r.get(f)) for r in rows if r.get(f) is not None]
        vals = [v for v in vals if v is not None]
        aggregates[f] = (sum(vals) / len(vals)) if vals else None

    for f in sum_fields:
        vals = [safe_int(r.get(f)) for r in rows if r.get(f) is not None]
        vals = [v for v in vals if v is not None]
        aggregates[f] = sum(vals) if vals else 0

    for f in str_fields:
        chosen = None
        best_ts = None
        for r in rows:
            val = r.get(f)
            if val:
                ts = None
                for ts_col in ("updated_at", "last_updated", "created_at", "processed_at"):
                    if r.get(ts_col):
                        try:
                            ts = datetime.fromisoformat(str(r.get(ts_col)).replace("Z", "+00:00"))
                        except Exception:
                            try:
                                ts = datetime.strptime(str(r.get(ts_col)), "%Y-%m-%d %H:%M:%S")
                            except Exception:
                                ts = None
                        if ts:
                            break
                if best_ts is None or (ts is not None and ts > best_ts):
                    best_ts = ts
                    chosen = val
        aggregates[f] = chosen

    last_updated = None
    for r in rows:
        for ts_col in ("updated_at", "last_updated", "processed_at", "created_at"):
            if r.get(ts_col):
                try:
                    cand = datetime.fromisoformat(str(r.get(ts_col)).replace("Z", "+00:00"))
                except Exception:
                    try:
                        cand = datetime.strptime(str(r.get(ts_col)), "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        cand = None
                if cand and (last_updated is None or cand > last_updated):
                    last_updated = cand

    return DailySignalsOut(
        signal_date=sig_date.isoformat(),
        capex_signal=aggregates.get("capex_signal"),
        energy_signal=aggregates.get("energy_signal"),
        compute_signal=aggregates.get("compute_signal"),
        depreciation_signal=aggregates.get("depreciation_signal"),
        thesis_balance=aggregates.get("thesis_balance"),
        veto_triggers=aggregates.get("veto_triggers"),
        signal_regime=aggregates.get("signal_regime"),
        notes=aggregates.get("notes"),
        rows_aggregated=len(rows),
        last_updated=last_updated.isoformat() if last_updated else None,
    )


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
    id: int
    article_hash: Optional[str] = None
    source_type: Optional[str] = None
    source_name: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
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


def parse_iso(dt_str: str) -> datetime:
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
        processed_at=to_display_tz(row.get("processed_at")),
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


@app.get("/processed_articles", response_model=PaginatedResponse)
def get_processed_articles(
    hours: Optional[int] = Query(None, ge=0, description="Articles from last N hours"),
    after: Optional[str] = Query(None, description="Articles after this date (YYYY-MM-DD or ISO datetime)"),
    before: Optional[str] = Query(None, description="Articles before this date (YYYY-MM-DD or ISO datetime)"),
    date: Optional[str] = Query(None, description="Specific date in EST/EDT (YYYY-MM-DD)"),
    time_window: Optional[List[str]] = Query(None, description="Time window(s) in HH:MM-HH:MM format (EST/EDT)"),
    category: Optional[str] = Query(None, description="Y2AI category"),
    impact_score_min: Optional[float] = Query(None, ge=0.0, le=1.0),
    sentiment: Optional[str] = Query(None, description="Comma-separated: bullish,bearish,neutral"),
    source_type: Optional[str] = Query(None),
    source_name: Optional[str] = Query(None),
    signal_type: Optional[str] = Query(None, description="Filter by signal: capex, energy, compute, depreciation, veto"),
    signal_detected: Optional[bool] = Query(None),
    include_in_weekly: Optional[bool] = Query(None),
    suggested_pillar: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    cutoff_start = None
    cutoff_end = None

    if date:
        # 🔥 FIX: Convert EST date to UTC range for query
        cutoff_start = convert_est_to_utc_for_query(date, is_start=True)
        cutoff_end = convert_est_to_utc_for_query(date, is_start=False)
    else:
        if hours is not None:
            # Hours are relative to current UTC time
            cutoff_start = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        if after:
            if len(after) == 10:  # YYYY-MM-DD format
                cutoff_start = convert_est_to_utc_for_query(after, is_start=True)
            else:  # ISO datetime
                parsed = parse_iso(after)
                cutoff_start = parsed.isoformat()

        if before:
            if len(before) == 10:  # YYYY-MM-DD format
                cutoff_end = convert_est_to_utc_for_query(before, is_start=False)
            else:  # ISO datetime
                parsed = parse_iso(before)
                cutoff_end = parsed.isoformat()

    # Time window filtering now works with EST/EDT times
    parsed_windows = []
    if time_window:
        import re

        def normalize_dashes(s: str) -> str:
            return (
                s.replace('\u2013', '-')
                .replace('\u2014', '-')
                .replace('\u2212', '-')
                .replace('\u2012', '-')
                .replace('\u2010', '-')
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

    # Query database with UTC times
    try:
        query = sb.table("processed_articles").select("*", count="exact")
        if cutoff_start:
            query = query.gte("processed_at", cutoff_start)
        if cutoff_end:
            query = query.lte("processed_at", cutoff_end)

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

        resp = query.order("processed_at", desc=True).execute()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    rows = resp.data or []

    # Convert UTC timestamps to EST/EDT for time window filtering
    def parse_processed_at_val(v):
        if v is None:
            return None

        if isinstance(v, datetime):
            dt = v
        else:
            s = str(v).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)

        # Ensure UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Convert to EST/EDT for time window comparison
        return dt.astimezone(DISPLAY_TZ)

    # Filter by time windows (in EST/EDT)
    filtered_rows = []
    if parsed_windows:
        for r in rows:
            p = parse_processed_at_val(r.get("processed_at"))
            if p is None:
                continue
            p_time = p.time()
            for (start_t, end_t) in parsed_windows:
                if start_t <= end_t:
                    if start_t <= p_time < end_t:
                        filtered_rows.append(r)
                        break
                else:
                    if p_time >= start_t or p_time < end_t:
                        filtered_rows.append(r)
                        break
    else:
        filtered_rows = rows

    total = len(filtered_rows)
    sliced = filtered_rows[offset: offset + limit]

    out_items = []
    for row in sliced:
        try:
            out_items.append(transform_to_nested(row))
        except Exception as e:
            print(f"Transform error for article {row.get('id')}: {e}")
            continue

    return PaginatedResponse(total=total, limit=limit, offset=offset, data=out_items)


@app.get("/signals/summary", response_model=SignalSummary)
def get_signal_summary(
    after: Optional[str] = Query(None),
    before: Optional[str] = Query(None),
    hours: Optional[int] = Query(168, ge=1),
):
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
    
    capex_count = sum(1 for r in data if r.get("capex_detected"))
    energy_count = sum(1 for r in data if r.get("energy_detected"))
    compute_count = sum(1 for r in data if r.get("compute_detected"))
    depreciation_count = sum(1 for r in data if r.get("depreciation_detected"))
    veto_count = sum(1 for r in data if r.get("veto_detected"))
    newsletter_count = sum(1 for r in data if r.get("include_in_weekly"))
    
    by_category = {}
    for r in data:
        cat = r.get("y2ai_category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    
    by_sentiment = {}
    for r in data:
        sent = r.get("sentiment", "unknown")
        by_sentiment[sent] = by_sentiment.get(sent, 0) + 1
    
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


@app.get("/health")
def health_check():
    try:
        sb.table("processed_articles").select("id").limit(1).execute()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)