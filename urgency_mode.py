"""
Urgency Mode Management - Y2AI ARGUS-1

Coordinates with Google Sheets via Supabase system_mode table.
Detects NLP anomalies and triggers urgency mode.
Reads urgency mode set by Sheets (flow anomalies).

Created: December 16, 2025
"""

import os
from datetime import datetime, timedelta, timezone
from supabase import create_client
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# NLP Urgency Thresholds (V2 - with warning support)
NLP_THRESHOLDS = {
    "burst_count": 20,
    "veto_triggers": 25,      # Full veto threshold
    "veto_warnings": 15,      # NEW: Warning threshold (elevated but not full urgency)
    "evi_score": 85,
    "thesis_swing": 15        # absolute change from previous day
}


def get_system_mode() -> dict:
    """
    Read current system mode from Supabase.
    Returns mode info or defaults to 'normal'.
    """
    try:
        response = supabase.table("system_mode")\
            .select("*")\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        
        if response.data:
            mode_data = response.data[0]
            
            # Check if urgency has expired
            if mode_data["mode"] == "urgency" and mode_data.get("end_time"):
                end_time = datetime.fromisoformat(mode_data["end_time"].replace("Z", "+00:00"))
                if datetime.now(timezone.utc) > end_time:
                    logger.info("Urgency mode expired, returning to normal")
                    deactivate_urgency_mode()
                    return {"mode": "normal", "triggered_by": None, "reason": None}
            
            return {
                "mode": mode_data["mode"],
                "triggered_by": mode_data.get("triggered_by"),
                "reason": mode_data.get("trigger_reason"),
                "end_time": mode_data.get("end_time")
            }
        
        return {"mode": "normal", "triggered_by": None, "reason": None}
    
    except Exception as e:
        logger.error(f"Error reading system mode: {e}")
        return {"mode": "normal", "triggered_by": None, "reason": None}


def activate_urgency_mode(triggered_by: str, reason: str, hours: int = 4):
    """
    Activate urgency mode in Supabase.
    
    Args:
        triggered_by: 'nlp', 'flows', or 'manual'
        reason: Description of what triggered urgency
        hours: Duration of urgency mode (default 4 hours)
    """
    end_time = datetime.now(timezone.utc) + timedelta(hours=hours)
    
    payload = {
        "mode": "urgency",
        "triggered_by": triggered_by,
        "trigger_reason": reason,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "end_time": end_time.isoformat()
    }
    
    try:
        supabase.table("system_mode").insert(payload).execute()
        logger.info(f"✓ Urgency mode activated: {reason}")
        logger.info(f"  Expires: {end_time.isoformat()}")
        return True
    except Exception as e:
        logger.error(f"Error activating urgency mode: {e}")
        return False


def deactivate_urgency_mode():
    """Deactivate urgency mode, return to normal."""
    payload = {
        "mode": "normal",
        "triggered_by": "auto_expire",
        "trigger_reason": "Urgency window elapsed",
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "end_time": None
    }
    
    try:
        supabase.table("system_mode").insert(payload).execute()
        logger.info("System mode returned to normal")
        return True
    except Exception as e:
        logger.error(f"Error deactivating urgency mode: {e}")
        return False


def get_today_signals() -> dict:
    """Fetch today's signals from daily_signals table."""
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        response = supabase.table("daily_signals")\
            .select("*")\
            .eq("date", today)\
            .execute()
        
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching today's signals: {e}")
        return None


def get_yesterday_signals() -> dict:
    """Fetch yesterday's signals for comparison."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        response = supabase.table("daily_signals")\
            .select("*")\
            .eq("date", yesterday)\
            .execute()
        
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching yesterday's signals: {e}")
        return None


def check_nlp_urgency_triggers() -> dict:
    """
    Check if NLP signals warrant urgency or warning mode.
    Called after NLP processing completes.
    
    Returns dict with:
        - triggered: bool (True if urgency was triggered)
        - warning: bool (True if warning state detected)
        - reason: str (description of triggers)
    """
    logger.info("=== CHECKING NLP URGENCY TRIGGERS (V2) ===")
    
    today = get_today_signals()
    yesterday = get_yesterday_signals()
    
    if not today:
        logger.warning("No signals data for today")
        return {"triggered": False, "warning": False, "reason": None}
    
    triggers = []
    warnings = []
    
    # Check burst count
    burst_count = today.get("burst_count") or 0
    if burst_count >= NLP_THRESHOLDS["burst_count"]:
        triggers.append(f"Burst count {burst_count} >= {NLP_THRESHOLDS['burst_count']}")
    logger.info(f"Burst Count: {burst_count}")
    
    # Check veto triggers (full urgency)
    veto_triggers = today.get("veto_triggers") or 0
    if veto_triggers >= NLP_THRESHOLDS["veto_triggers"]:
        triggers.append(f"Veto triggers {veto_triggers} >= {NLP_THRESHOLDS['veto_triggers']}")
    elif veto_triggers >= NLP_THRESHOLDS["veto_warnings"]:
        warnings.append(f"Veto warnings {veto_triggers} >= {NLP_THRESHOLDS['veto_warnings']}")
    logger.info(f"Veto Triggers: {veto_triggers}")
    
    # Check veto warnings count (NEW in V2)
    veto_warnings = today.get("veto_warnings") or 0
    if veto_warnings > 0:
        warnings.append(f"Warning-level vetos detected: {veto_warnings}")
    logger.info(f"Veto Warnings: {veto_warnings}")
    
    # Check EVI score
    evi_score = today.get("evi_score") or 0
    if evi_score >= NLP_THRESHOLDS["evi_score"]:
        triggers.append(f"EVI score {evi_score} >= {NLP_THRESHOLDS['evi_score']}")
    elif evi_score >= 75:  # Warning threshold for EVI
        warnings.append(f"EVI elevated: {evi_score}")
    logger.info(f"EVI Score: {evi_score}")
    
    # Check thesis swing (vs yesterday)
    if yesterday:
        today_thesis = today.get("thesis_balance") or 0
        yesterday_thesis = yesterday.get("thesis_balance") or 0
        thesis_swing = abs(today_thesis - yesterday_thesis)
        
        if thesis_swing >= NLP_THRESHOLDS["thesis_swing"]:
            triggers.append(f"Thesis swing {thesis_swing:.1f} >= {NLP_THRESHOLDS['thesis_swing']}")
        logger.info(f"Thesis Swing: {thesis_swing:.1f}")
    
    # Activate urgency if any full triggers
    if triggers:
        reason = "; ".join(triggers)
        logger.warning(f"⚠️ NLP URGENCY TRIGGERED: {reason}")
        activate_urgency_mode("nlp", reason)
        return {"triggered": True, "warning": False, "reason": reason}
    
    # Log warnings but don't activate urgency
    if warnings:
        warning_reason = "; ".join(warnings)
        logger.info(f"⚡ NLP WARNING STATE: {warning_reason}")
        return {"triggered": False, "warning": True, "reason": warning_reason}
    
    logger.info("✓ NLP triggers clear")
    return {"triggered": False, "warning": False, "reason": None}


def is_urgency_mode() -> bool:
    """Quick check if currently in urgency mode."""
    mode = get_system_mode()
    return mode["mode"] == "urgency"


def run_with_urgency_check(normal_func, urgency_func=None):
    """
    Wrapper to run appropriate function based on system mode.
    
    Args:
        normal_func: Function to run in normal mode
        urgency_func: Function to run in urgency mode (defaults to normal_func)
    """
    mode = get_system_mode()
    
    if mode["mode"] == "urgency":
        logger.info(f"🚨 URGENCY MODE ACTIVE")
        logger.info(f"   Triggered by: {mode['triggered_by']}")
        logger.info(f"   Reason: {mode['reason']}")
        
        if urgency_func:
            return urgency_func()
        else:
            return normal_func()
    else:
        logger.info("✓ Normal mode")
        return normal_func()


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python urgency_mode.py [check|activate|deactivate|status]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "check":
        check_nlp_urgency_triggers()
    
    elif command == "activate":
        reason = sys.argv[2] if len(sys.argv) > 2 else "Manual activation"
        activate_urgency_mode("manual", reason)
    
    elif command == "deactivate":
        deactivate_urgency_mode()
    
    elif command == "status":
        mode = get_system_mode()
        print(f"Current mode: {mode['mode']}")
        if mode['triggered_by']:
            print(f"Triggered by: {mode['triggered_by']}")
            print(f"Reason: {mode['reason']}")
            print(f"Expires: {mode.get('end_time', 'N/A')}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)