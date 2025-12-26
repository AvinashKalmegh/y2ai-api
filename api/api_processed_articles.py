# api_processed_articles.py
"""
Y2AI Processed Articles API
Returns articles with nested signal structure for clean frontend consumption
Includes ARGUS-1 regime signals, bubble index, stock tracker, and hypergraph endpoints
"""

import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Any, Union

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client
from fastapi.middleware.cors import CORSMiddleware
from zoneinfo import ZoneInfo
from datetime import date as date_class

DISPLAY_TZ = ZoneInfo("America/New_York")  # EST/EDT


def convert_est_to_utc_for_query(date_str: str, is_start: bool = True):
    """
    Convert EST/EDT date string to UTC for Supabase queries.
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    if is_start:
        local_dt = datetime.combine(date_obj, datetime.min.time())
    else:
        local_dt = datetime.combine(date_obj, datetime.max.time())
    
    local_dt = local_dt.replace(tzinfo=DISPLAY_TZ)
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

app = FastAPI(title="Y2AI API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", 'https://y2ai-frontend.vercel.app'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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


def safe_list(val) -> Optional[List[str]]:
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
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


def parse_ymd(date_str: Optional[str]) -> date_class:
    if date_str is None:
        return datetime.utcnow().date()
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")


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


# =============================================================================
# ARGUS-1 MASTER SIGNALS MODELS
# =============================================================================

class AMRIOut(BaseModel):
    composite: Optional[float] = None
    amri_s: Optional[float] = None
    amri_b: Optional[float] = None
    amri_c: Optional[float] = None


class TTIOut(BaseModel):
    display: Optional[str] = None
    rate: Optional[float] = None
    days_to_fragile: Optional[int] = None
    days_to_break: Optional[int] = None


class SACOut(BaseModel):
    composite: Optional[float] = None
    weakest_link: Optional[str] = None


class ContagionOut(BaseModel):
    score: Optional[float] = None
    regime: Optional[str] = None


class FingerprintOut(BaseModel):
    episode: Optional[str] = None
    match_score: Optional[float] = None
    pattern: Optional[str] = None


class RotationOut(BaseModel):
    leader: Optional[str] = None
    laggard: Optional[str] = None
    spread: Optional[float] = None


class EventsOut(BaseModel):
    next_event: Optional[str] = None
    days_to_event: Optional[int] = None


class RecoveryOut(BaseModel):
    active: Optional[bool] = None
    strength: Optional[float] = None


class ARGUS1SignalOut(BaseModel):
    date: str
    calculated_at: Optional[str] = None
    
    # Core regime
    regime: Optional[str] = None
    authority: Optional[str] = None
    confidence: Optional[str] = None
    
    # Components
    amri: AMRIOut = AMRIOut()
    tti: TTIOut = TTIOut()
    sac: SACOut = SACOut()
    contagion: ContagionOut = ContagionOut()
    fingerprint: FingerprintOut = FingerprintOut()
    rotation: RotationOut = RotationOut()
    events: EventsOut = EventsOut()
    recovery: RecoveryOut = RecoveryOut()
    
    # VETO
    veto_active: Optional[bool] = None
    veto_count: Optional[int] = None
    thesis_balance: Optional[float] = None


class ARGUS1HistoryResponse(BaseModel):
    total: int
    data: List[ARGUS1SignalOut]


# =============================================================================
# ARGUS-1 ENDPOINTS
# =============================================================================

@app.get("/argus1/latest", response_model=ARGUS1SignalOut)
def get_argus1_latest():
    """
    Get the latest ARGUS-1 regime signal.
    Returns the most recent calculation with all components.
    """
    try:
        resp = sb.table("argus_master_signals")\
            .select("*")\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        rows = resp.data or []
        
        if not rows:
            raise HTTPException(status_code=404, detail="No ARGUS-1 signals found")
        
        row = rows[0]
        
        return ARGUS1SignalOut(
            date=str(row.get("date", "")),
            calculated_at=row.get("calculated_at"),
            regime=row.get("regime"),
            authority=row.get("authority"),
            confidence=row.get("confidence"),
            amri=AMRIOut(
                composite=row.get("amri_composite"),
                amri_s=row.get("amri_s"),
                amri_b=row.get("amri_b"),
                amri_c=row.get("amri_c"),
            ),
            tti=TTIOut(
                display=row.get("tti_display"),
                rate=row.get("tti_rate"),
                days_to_fragile=row.get("tti_days_to_fragile"),
                days_to_break=row.get("tti_days_to_break"),
            ),
            sac=SACOut(
                composite=row.get("sac_composite"),
                weakest_link=row.get("sac_weakest"),
            ),
            contagion=ContagionOut(
                score=row.get("contagion_score"),
                regime=row.get("contagion_regime"),
            ),
            fingerprint=FingerprintOut(
                episode=row.get("fingerprint_episode"),
                match_score=row.get("fingerprint_match"),
                pattern=row.get("fingerprint_pattern"),
            ),
            rotation=RotationOut(
                leader=row.get("rotation_leader"),
                laggard=row.get("rotation_laggard"),
                spread=row.get("rotation_spread"),
            ),
            events=EventsOut(
                next_event=row.get("next_event"),
                days_to_event=row.get("days_to_event"),
            ),
            recovery=RecoveryOut(
                active=row.get("recovery_active"),
                strength=row.get("recovery_strength"),
            ),
            veto_active=row.get("veto_active"),
            veto_count=row.get("veto_count"),
            thesis_balance=row.get("thesis_balance"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/argus1/history", response_model=ARGUS1HistoryResponse)
def get_argus1_history(
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    after: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    before: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Get ARGUS-1 signal history for charting regime changes over time.
    """
    try:
        query = sb.table("argus_master_signals").select("*")
        
        if after:
            query = query.gte("date", after)
        if before:
            query = query.lte("date", before)
        
        if not after and not before:
            cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            query = query.gte("date", cutoff)
        
        query = query.order("date", desc=False)
        
        resp = query.execute()
        rows = resp.data or []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
    out = []
    for row in rows:
        out.append(ARGUS1SignalOut(
            date=str(row.get("date", "")),
            calculated_at=row.get("calculated_at"),
            regime=row.get("regime"),
            authority=row.get("authority"),
            confidence=row.get("confidence"),
            amri=AMRIOut(
                composite=row.get("amri_composite"),
                amri_s=row.get("amri_s"),
                amri_b=row.get("amri_b"),
                amri_c=row.get("amri_c"),
            ),
            tti=TTIOut(
                display=row.get("tti_display"),
                rate=row.get("tti_rate"),
                days_to_fragile=row.get("tti_days_to_fragile"),
                days_to_break=row.get("tti_days_to_break"),
            ),
            sac=SACOut(
                composite=row.get("sac_composite"),
                weakest_link=row.get("sac_weakest"),
            ),
            contagion=ContagionOut(
                score=row.get("contagion_score"),
                regime=row.get("contagion_regime"),
            ),
            fingerprint=FingerprintOut(
                episode=row.get("fingerprint_episode"),
                match_score=row.get("fingerprint_match"),
                pattern=row.get("fingerprint_pattern"),
            ),
            rotation=RotationOut(
                leader=row.get("rotation_leader"),
                laggard=row.get("rotation_laggard"),
                spread=row.get("rotation_spread"),
            ),
            events=EventsOut(
                next_event=row.get("next_event"),
                days_to_event=row.get("days_to_event"),
            ),
            recovery=RecoveryOut(
                active=row.get("recovery_active"),
                strength=row.get("recovery_strength"),
            ),
            veto_active=row.get("veto_active"),
            veto_count=row.get("veto_count"),
            thesis_balance=row.get("thesis_balance"),
        ))
    
    return ARGUS1HistoryResponse(total=len(out), data=out)


@app.get("/argus1/regime-summary")
def get_regime_summary(days: int = Query(30, ge=1, le=365)):
    """
    Get regime distribution summary for the specified period.
    Useful for pie charts or regime frequency analysis.
    """
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        resp = sb.table("argus_master_signals")\
            .select("date, regime, authority, amri_composite, veto_active")\
            .gte("date", cutoff)\
            .order("date", desc=False)\
            .execute()
        
        rows = resp.data or []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
    regime_counts = {}
    authority_counts = {}
    veto_days = 0
    amri_values = []
    
    for row in rows:
        regime = row.get("regime", "UNKNOWN")
        authority = row.get("authority", "UNKNOWN")
        
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        authority_counts[authority] = authority_counts.get(authority, 0) + 1
        
        if row.get("veto_active"):
            veto_days += 1
        
        if row.get("amri_composite") is not None:
            amri_values.append(row.get("amri_composite"))
    
    return {
        "period_days": days,
        "total_signals": len(rows),
        "regime_distribution": regime_counts,
        "authority_distribution": authority_counts,
        "veto_days": veto_days,
        "veto_percentage": round(veto_days / len(rows) * 100, 1) if rows else 0,
        "amri_avg": round(sum(amri_values) / len(amri_values), 1) if amri_values else None,
        "amri_min": min(amri_values) if amri_values else None,
        "amri_max": max(amri_values) if amri_values else None,
    }


# =============================================================================
# BUBBLE INDEX ENDPOINT
# =============================================================================

class BubbleIndexOut(BaseModel):
    date: str
    vix: Optional[float] = None
    cape: Optional[float] = None
    credit_spread_ig: Optional[float] = None
    credit_spread_hy: Optional[float] = None
    bubble_index: Optional[float] = None
    bifurcation_score: Optional[float] = None
    regime: Optional[str] = None
    vix_zscore: Optional[float] = None
    cape_zscore: Optional[float] = None
    credit_zscore: Optional[float] = None


class BubbleIndexHistoryResponse(BaseModel):
    total: int
    data: List[BubbleIndexOut]


@app.get("/bubble-index/latest", response_model=BubbleIndexOut)
def get_bubble_index_latest():
    """Get the latest bubble index reading."""
    try:
        resp = sb.table("bubble_index_daily")\
            .select("*")\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        rows = resp.data or []
        
        if not rows:
            raise HTTPException(status_code=404, detail="No bubble index data found")
        
        row = rows[0]
        
        return BubbleIndexOut(
            date=str(row.get("date", "")),
            vix=row.get("vix"),
            cape=row.get("cape"),
            credit_spread_ig=row.get("credit_spread_ig"),
            credit_spread_hy=row.get("credit_spread_hy"),
            bubble_index=row.get("bubble_index"),
            bifurcation_score=row.get("bifurcation_score"),
            regime=row.get("regime"),
            vix_zscore=row.get("vix_zscore"),
            cape_zscore=row.get("cape_zscore"),
            credit_zscore=row.get("credit_zscore"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/bubble-index/history", response_model=BubbleIndexHistoryResponse)
def get_bubble_index_history(
    days: int = Query(90, ge=1, le=365, description="Number of days of history"),
    after: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    before: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """Get bubble index history for charting."""
    try:
        query = sb.table("bubble_index_daily").select("*")
        
        if after:
            query = query.gte("date", after)
        if before:
            query = query.lte("date", before)
        
        if not after and not before:
            cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            query = query.gte("date", cutoff)
        
        query = query.order("date", desc=False)
        
        resp = query.execute()
        rows = resp.data or []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
    out = []
    for row in rows:
        out.append(BubbleIndexOut(
            date=str(row.get("date", "")),
            vix=row.get("vix"),
            cape=row.get("cape"),
            credit_spread_ig=row.get("credit_spread_ig"),
            credit_spread_hy=row.get("credit_spread_hy"),
            bubble_index=row.get("bubble_index"),
            bifurcation_score=row.get("bifurcation_score"),
            regime=row.get("regime"),
            vix_zscore=row.get("vix_zscore"),
            cape_zscore=row.get("cape_zscore"),
            credit_zscore=row.get("credit_zscore"),
        ))
    
    return BubbleIndexHistoryResponse(total=len(out), data=out)


# =============================================================================
# STOCK TRACKER ENDPOINT
# =============================================================================

class PillarOut(BaseModel):
    name: Optional[str] = None
    pillar_id: Optional[str] = None
    stocks: Optional[List[str]] = None
    avg_today: Optional[float] = None
    avg_5day: Optional[float] = None
    avg_ytd: Optional[float] = None


class StockOut(BaseModel):
    ticker: Optional[str] = None
    name: Optional[str] = None
    pillar: Optional[str] = None
    price: Optional[float] = None
    change_today: Optional[float] = None
    change_5day: Optional[float] = None
    change_ytd: Optional[float] = None


class StockTrackerOut(BaseModel):
    date: str
    y2ai_index_today: Optional[float] = None
    y2ai_index_5day: Optional[float] = None
    y2ai_index_ytd: Optional[float] = None
    spy_today: Optional[float] = None
    spy_5day: Optional[float] = None
    spy_ytd: Optional[float] = None
    qqq_today: Optional[float] = None
    qqq_5day: Optional[float] = None
    qqq_ytd: Optional[float] = None
    status: Optional[str] = None
    best_stock: Optional[str] = None
    worst_stock: Optional[str] = None
    best_pillar: Optional[str] = None
    worst_pillar: Optional[str] = None
    pillars: Optional[List[PillarOut]] = None
    stocks: Optional[List[StockOut]] = None


@app.get("/stock-tracker/latest", response_model=StockTrackerOut)
def get_stock_tracker_latest():
    """Get the latest stock tracker reading with full pillar and stock details."""
    try:
        resp = sb.table("stock_tracker_daily")\
            .select("*")\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        rows = resp.data or []
        
        if not rows:
            raise HTTPException(status_code=404, detail="No stock tracker data found")
        
        row = rows[0]
        
        # Parse pillars JSON
        pillars_raw = row.get("pillars")
        pillars = []
        if pillars_raw:
            if isinstance(pillars_raw, str):
                pillars_raw = json.loads(pillars_raw)
            for p in pillars_raw:
                pillars.append(PillarOut(
                    name=p.get("name"),
                    pillar_id=p.get("pillar_id"),
                    stocks=p.get("stocks"),
                    avg_today=p.get("avg_today"),
                    avg_5day=p.get("avg_5day"),
                    avg_ytd=p.get("avg_ytd"),
                ))
        
        # Parse stocks JSON
        stocks_raw = row.get("stocks")
        stocks = []
        if stocks_raw:
            if isinstance(stocks_raw, str):
                stocks_raw = json.loads(stocks_raw)
            for s in stocks_raw:
                stocks.append(StockOut(
                    ticker=s.get("ticker"),
                    name=s.get("name"),
                    pillar=s.get("pillar"),
                    price=s.get("price"),
                    change_today=s.get("change_today"),
                    change_5day=s.get("change_5day"),
                    change_ytd=s.get("change_ytd"),
                ))
        
        return StockTrackerOut(
            date=str(row.get("date", "")),
            y2ai_index_today=row.get("y2ai_index_today"),
            y2ai_index_5day=row.get("y2ai_index_5day"),
            y2ai_index_ytd=row.get("y2ai_index_ytd"),
            spy_today=row.get("spy_today"),
            spy_5day=row.get("spy_5day"),
            spy_ytd=row.get("spy_ytd"),
            qqq_today=row.get("qqq_today"),
            qqq_5day=row.get("qqq_5day"),
            qqq_ytd=row.get("qqq_ytd"),
            status=row.get("status"),
            best_stock=row.get("best_stock"),
            worst_stock=row.get("worst_stock"),
            best_pillar=row.get("best_pillar"),
            worst_pillar=row.get("worst_pillar"),
            pillars=pillars,
            stocks=stocks,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


# =============================================================================
# HYPERGRAPH ENDPOINT
# =============================================================================

class HypergraphOut(BaseModel):
    date: str
    hyperedge_count: Optional[int] = None
    avg_hyperedge_size: Optional[float] = None
    max_hyperedge_size: Optional[int] = None
    cross_pillar_count: Optional[int] = None
    cross_pillar_ratio: Optional[float] = None
    stability_score: Optional[float] = None
    contagion_score: Optional[float] = None
    regime: Optional[str] = None
    bridge_stocks: Optional[List[str]] = None


class HypergraphHistoryResponse(BaseModel):
    total: int
    data: List[HypergraphOut]


@app.get("/hypergraph/latest", response_model=HypergraphOut)
def get_hypergraph_latest():
    """Get the latest hypergraph/contagion reading."""
    try:
        resp = sb.table("hypergraph_signals")\
            .select("*")\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        rows = resp.data or []
        
        if not rows:
            raise HTTPException(status_code=404, detail="No hypergraph data found")
        
        row = rows[0]
        
        bridge = row.get("bridge_stocks")
        if isinstance(bridge, str):
            try:
                bridge = json.loads(bridge)
            except:
                bridge = []
        
        return HypergraphOut(
            date=str(row.get("date", "")),
            hyperedge_count=row.get("hyperedge_count"),
            avg_hyperedge_size=row.get("avg_hyperedge_size"),
            max_hyperedge_size=row.get("max_hyperedge_size"),
            cross_pillar_count=row.get("cross_pillar_count"),
            cross_pillar_ratio=row.get("cross_pillar_ratio"),
            stability_score=row.get("stability_score"),
            contagion_score=row.get("contagion_score"),
            regime=row.get("regime"),
            bridge_stocks=bridge,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/hypergraph/history", response_model=HypergraphHistoryResponse)
def get_hypergraph_history(
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    after: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    before: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """Get hypergraph history for charting contagion over time."""
    try:
        query = sb.table("hypergraph_signals").select("*")
        
        if after:
            query = query.gte("date", after)
        if before:
            query = query.lte("date", before)
        
        if not after and not before:
            cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            query = query.gte("date", cutoff)
        
        query = query.order("date", desc=False)
        
        resp = query.execute()
        rows = resp.data or []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
    out = []
    for row in rows:
        bridge = row.get("bridge_stocks")
        if isinstance(bridge, str):
            try:
                bridge = json.loads(bridge)
            except:
                bridge = []
        
        out.append(HypergraphOut(
            date=str(row.get("date", "")),
            hyperedge_count=row.get("hyperedge_count"),
            avg_hyperedge_size=row.get("avg_hyperedge_size"),
            max_hyperedge_size=row.get("max_hyperedge_size"),
            cross_pillar_count=row.get("cross_pillar_count"),
            cross_pillar_ratio=row.get("cross_pillar_ratio"),
            stability_score=row.get("stability_score"),
            contagion_score=row.get("contagion_score"),
            regime=row.get("regime"),
            bridge_stocks=bridge,
        ))
    
    return HypergraphHistoryResponse(total=len(out), data=out)


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
        
        if after:
            query = query.gte("date", after)
        if before:
            query = query.lte("date", before)
        
        if not after and not before:
            cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            query = query.gte("date", cutoff)
        
        query = query.order("date", desc=False)
        
        resp = query.execute()
        rows = resp.data or []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
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
    """Insert or update a single day's metrics."""
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
    """Bulk insert/update metrics history."""
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
# DAILY SIGNALS ENDPOINT
# =============================================================================

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
    
    # NLP Signals
    nci_score: Optional[float] = None
    nci_regime: Optional[str] = None
    nci_top_category: Optional[str] = None
    nci_top_keyword: Optional[str] = None
    
    npd_score: Optional[float] = None
    npd_regime: Optional[str] = None
    npd_short_avg: Optional[float] = None
    npd_long_avg: Optional[float] = None
    
    burst_count: Optional[int] = None
    burst_regime: Optional[str] = None
    burst_top_keyword: Optional[str] = None
    burst_top_ratio: Optional[float] = None
    
    evi_score: Optional[float] = None
    evi_regime: Optional[str] = None
    evi_volume: Optional[float] = None
    evi_velocity: Optional[float] = None
    evi_urgency: Optional[float] = None


@app.get("/daily-signals", response_model=DailySignalsOut)
def get_daily_signals(date: Optional[str] = Query(None, description="Date in YYYY-MM-DD (defaults to today UTC)")):
    """Fetch aggregated daily signals for a particular date."""
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
            nci_score=None, nci_regime=None, nci_top_category=None, nci_top_keyword=None,
            npd_score=None, npd_regime=None, npd_short_avg=None, npd_long_avg=None,
            burst_count=None, burst_regime=None, burst_top_keyword=None, burst_top_ratio=None,
            evi_score=None, evi_regime=None, evi_volume=None, evi_velocity=None, evi_urgency=None,
        )

    avg_fields = ["capex_signal", "energy_signal", "compute_signal", "depreciation_signal", "thesis_balance"]
    sum_fields = ["veto_triggers"]
    str_fields = ["signal_regime", "notes"]

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
        
        nci_score=safe_float(rows[0].get("nci_score")) if rows else None,
        nci_regime=rows[0].get("nci_regime") if rows else None,
        nci_top_category=rows[0].get("nci_top_category") if rows else None,
        nci_top_keyword=rows[0].get("nci_top_keyword") if rows else None,
        
        npd_score=safe_float(rows[0].get("npd_score")) if rows else None,
        npd_regime=rows[0].get("npd_regime") if rows else None,
        npd_short_avg=safe_float(rows[0].get("npd_short_avg")) if rows else None,
        npd_long_avg=safe_float(rows[0].get("npd_long_avg")) if rows else None,
        
        burst_count=safe_int(rows[0].get("burst_count")) if rows else None,
        burst_regime=rows[0].get("burst_regime") if rows else None,
        burst_top_keyword=rows[0].get("burst_top_keyword") if rows else None,
        burst_top_ratio=safe_float(rows[0].get("burst_top_ratio")) if rows else None,
        
        evi_score=safe_float(rows[0].get("evi_score")) if rows else None,
        evi_regime=rows[0].get("evi_regime") if rows else None,
        evi_volume=safe_float(rows[0].get("evi_volume")) if rows else None,
        evi_velocity=safe_float(rows[0].get("evi_velocity")) if rows else None,
        evi_urgency=safe_float(rows[0].get("evi_urgency")) if rows else None,
    )


# =============================================================================
# PROCESSED ARTICLES MODELS
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


def transform_to_nested(row: dict) -> ArticleOut:
    return ArticleOut(
        id=row.get("id"),
        article_hash=row.get("article_hash"),
        source_type=row.get("source_type"),
        source_name=row.get("source_name"),
        title=row.get("title"),
        url=row.get("url"),
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
        cutoff_start = convert_est_to_utc_for_query(date, is_start=True)
        cutoff_end = convert_est_to_utc_for_query(date, is_start=False)
    else:
        if hours is not None:
            cutoff_start = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        if after:
            if len(after) == 10:
                cutoff_start = convert_est_to_utc_for_query(after, is_start=True)
            else:
                parsed = parse_iso(after)
                cutoff_start = parsed.isoformat()

        if before:
            if len(before) == 10:
                cutoff_end = convert_est_to_utc_for_query(before, is_start=False)
            else:
                parsed = parse_iso(before)
                cutoff_end = parsed.isoformat()

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

    def parse_processed_at_val(v):
        if v is None:
            return None

        if isinstance(v, datetime):
            dt = v
        else:
            s = str(v).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(DISPLAY_TZ)

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


# =============================================================================
# DASHBOARD SUMMARY ENDPOINT
# =============================================================================

@app.get("/dashboard/summary")
def get_dashboard_summary():
    """
    Get a complete dashboard summary with latest data from all sources.
    Combines ARGUS-1, bubble index, stock tracker, hypergraph, and daily signals.
    """
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "argus1": None,
        "bubble_index": None,
        "stock_tracker": None,
        "hypergraph": None,
        "daily_signals": None,
    }
    
    try:
        # ARGUS-1
        resp = sb.table("argus_master_signals").select("*").order("date", desc=True).limit(1).execute()
        if resp.data:
            row = resp.data[0]
            result["argus1"] = {
                "date": row.get("date"),
                "regime": row.get("regime"),
                "authority": row.get("authority"),
                "amri": row.get("amri_composite"),
                "tti_display": row.get("tti_display"),
                "sac": row.get("sac_composite"),
                "sac_weakest": row.get("sac_weakest"),
                "veto_active": row.get("veto_active"),
                "contagion": row.get("contagion_score"),
                "fingerprint": row.get("fingerprint_episode"),
            }
    except:
        pass
    
    try:
        # Bubble Index
        resp = sb.table("bubble_index_daily").select("*").order("date", desc=True).limit(1).execute()
        if resp.data:
            row = resp.data[0]
            result["bubble_index"] = {
                "date": row.get("date"),
                "vix": row.get("vix"),
                "bubble_index": row.get("bubble_index"),
                "bifurcation_score": row.get("bifurcation_score"),
                "regime": row.get("regime"),
            }
    except:
        pass
    
    try:
        # Stock Tracker
        resp = sb.table("stock_tracker_daily").select("*").order("date", desc=True).limit(1).execute()
        if resp.data:
            row = resp.data[0]
            result["stock_tracker"] = {
                "date": row.get("date"),
                "y2ai_index_today": row.get("y2ai_index_today"),
                "y2ai_index_ytd": row.get("y2ai_index_ytd"),
                "spy_today": row.get("spy_today"),
                "status": row.get("status"),
                "best_pillar": row.get("best_pillar"),
                "worst_pillar": row.get("worst_pillar"),
            }
    except:
        pass
    
    try:
        # Hypergraph
        resp = sb.table("hypergraph_signals").select("*").order("date", desc=True).limit(1).execute()
        if resp.data:
            row = resp.data[0]
            result["hypergraph"] = {
                "date": row.get("date"),
                "contagion_score": row.get("contagion_score"),
                "regime": row.get("regime"),
                "cross_pillar_ratio": row.get("cross_pillar_ratio"),
                "stability_score": row.get("stability_score"),
            }
    except:
        pass
    
    try:
        # Daily Signals
        resp = sb.table("daily_signals").select("*").order("date", desc=True).limit(1).execute()
        if resp.data:
            row = resp.data[0]
            result["daily_signals"] = {
                "date": row.get("date"),
                "veto_triggers": row.get("veto_triggers"),
                "signal_regime": row.get("signal_regime"),
                "thesis_balance": row.get("thesis_balance"),
                "nci_score": row.get("nci_score"),
                "evi_score": row.get("evi_score"),
                "burst_count": row.get("burst_count"),
            }
    except:
        pass
    
    return result


# =============================================================================
# HEALTH CHECK
# =============================================================================

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
