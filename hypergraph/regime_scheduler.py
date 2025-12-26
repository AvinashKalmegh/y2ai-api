"""
Hypergraph Regime Scheduler - Y2AI ARGUS-1

Determines whether hypergraph should run based on:
1. Current urgency mode (from system_mode table)
2. Current contagion regime (from hypergraph_signals table)
3. Time since last update (staleness check)

Thresholds:
- NORMAL regime: update every 24h
- ELEVATED (contagion 50-70): update every 12h  
- CONTAGION (contagion >70) or VETO: update every 6h
- BREAK or urgency mode: update every 2h

Created: December 2025
"""

import os
from datetime import datetime, timezone, timedelta
from supabase import create_client
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Update intervals in hours based on regime
UPDATE_INTERVALS = {
    "STABLE": 24,
    "TRANSITIONING": 12,
    "FRAGMENTING": 8,
    "CONTAGION": 6,
    "BREAK": 2,
    "URGENCY": 2,  # When urgency mode is active
}

# Staleness thresholds
STALENESS_WARNING_HOURS = 24
STALENESS_CRITICAL_HOURS = 48


def get_latest_hypergraph() -> dict:
    """Fetch most recent hypergraph signal from Supabase."""
    try:
        response = supabase.table("hypergraph_signals")\
            .select("*")\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching hypergraph data: {e}")
        return None


def get_system_mode() -> str:
    """Check if system is in urgency mode."""
    try:
        response = supabase.table("system_mode")\
            .select("mode, end_time")\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        
        if response.data:
            mode_data = response.data[0]
            if mode_data["mode"] == "urgency":
                # Check if expired
                if mode_data.get("end_time"):
                    end_time = datetime.fromisoformat(
                        mode_data["end_time"].replace("Z", "+00:00")
                    )
                    if datetime.now(timezone.utc) > end_time:
                        return "normal"
                return "urgency"
        return "normal"
    except Exception as e:
        logger.error(f"Error checking system mode: {e}")
        return "normal"


def calculate_staleness(last_update: str) -> float:
    """Calculate hours since last update."""
    try:
        # Handle both date and datetime formats
        if "T" in last_update:
            last_dt = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
        else:
            # Just a date, assume end of day UTC
            last_dt = datetime.strptime(last_update, "%Y-%m-%d")
            last_dt = last_dt.replace(hour=22, minute=30, tzinfo=timezone.utc)
        
        now = datetime.now(timezone.utc)
        delta = now - last_dt
        return delta.total_seconds() / 3600
    except Exception as e:
        logger.error(f"Error calculating staleness: {e}")
        return 999  # Force update on error


def should_update_hypergraph() -> dict:
    """
    Determine if hypergraph should be updated now.
    
    Returns dict with:
        - should_run: bool
        - reason: str
        - staleness_hours: float
        - current_regime: str
        - required_interval: int
    """
    result = {
        "should_run": False,
        "reason": "",
        "staleness_hours": 0,
        "current_regime": "UNKNOWN",
        "required_interval": 24
    }
    
    # Check urgency mode first
    system_mode = get_system_mode()
    if system_mode == "urgency":
        result["current_regime"] = "URGENCY"
        result["required_interval"] = UPDATE_INTERVALS["URGENCY"]
        logger.info("🚨 System in URGENCY mode - 2h update interval")
    
    # Get latest hypergraph data
    latest = get_latest_hypergraph()
    
    if not latest:
        result["should_run"] = True
        result["reason"] = "No hypergraph data found"
        result["staleness_hours"] = 999
        return result
    
    # Calculate staleness
    last_date = latest.get("date") or latest.get("created_at")
    staleness = calculate_staleness(last_date)
    result["staleness_hours"] = staleness
    
    # Determine regime from contagion score
    contagion = latest.get("contagion_score", 0)
    regime = latest.get("regime", "STABLE")
    
    if system_mode != "urgency":
        result["current_regime"] = regime
        result["required_interval"] = UPDATE_INTERVALS.get(regime, 24)
    
    logger.info(f"Current regime: {result['current_regime']}")
    logger.info(f"Contagion score: {contagion}")
    logger.info(f"Staleness: {staleness:.1f} hours")
    logger.info(f"Required interval: {result['required_interval']} hours")
    
    # Check if update needed
    if staleness >= STALENESS_CRITICAL_HOURS:
        result["should_run"] = True
        result["reason"] = f"CRITICAL: Data {staleness:.1f}h stale (>{STALENESS_CRITICAL_HOURS}h)"
    elif staleness >= result["required_interval"]:
        result["should_run"] = True
        result["reason"] = f"Scheduled: {staleness:.1f}h since last update (interval: {result['required_interval']}h)"
    elif staleness >= STALENESS_WARNING_HOURS:
        result["should_run"] = True
        result["reason"] = f"WARNING: Data {staleness:.1f}h stale"
    else:
        result["reason"] = f"Fresh: {staleness:.1f}h old (next update in {result['required_interval'] - staleness:.1f}h)"
    
    return result


def log_staleness_to_supabase(staleness_hours: float, regime: str):
    """Log staleness check to monitoring table (optional)."""
    try:
        supabase.table("hypergraph_health").insert({
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "staleness_hours": staleness_hours,
            "regime": regime,
            "status": "critical" if staleness_hours > STALENESS_CRITICAL_HOURS 
                     else "warning" if staleness_hours > STALENESS_WARNING_HOURS 
                     else "ok"
        }).execute()
    except Exception as e:
        # Table might not exist, that's fine
        logger.debug(f"Could not log to hypergraph_health: {e}")


def main():
    """Check if hypergraph should run and return exit code."""
    logger.info("=" * 50)
    logger.info("HYPERGRAPH REGIME SCHEDULER")
    logger.info("=" * 50)
    
    result = should_update_hypergraph()
    
    logger.info("")
    logger.info(f"Decision: {'RUN' if result['should_run'] else 'SKIP'}")
    logger.info(f"Reason: {result['reason']}")
    
    # Print parseable output for GitHub Actions
    should_run = 'true' if result['should_run'] else 'false'
    print(f"SHOULD_RUN={should_run}")
    print(f"REGIME={result['current_regime']}")
    print(f"STALENESS={result['staleness_hours']:.1f}h")
    
    # Log to Supabase for monitoring
    log_staleness_to_supabase(result['staleness_hours'], result['current_regime'])
    
    if result['should_run']:
        logger.info("")
        logger.info("✓ Proceeding with hypergraph update...")
        return 0  # Success - run hypergraph
    else:
        logger.info("")
        logger.info("⏭️  Skipping hypergraph update")
        return 1  # Skip - don't run hypergraph


if __name__ == "__main__":
    import sys
    sys.exit(main())