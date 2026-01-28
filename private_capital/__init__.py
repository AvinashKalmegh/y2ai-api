"""
Private Capital Tracker Module for ARGUS-1
==========================================

Tracks AI venture capital and private equity funding as a leading indicator.

Usage:
    from private_capital import PrivateCapitalTracker, update_private_capital
    
    # Direct use
    tracker = PrivateCapitalTracker(supabase_client=client)
    result = tracker.update_dial()
    
    # Workflow integration (for orchestrator)
    result = update_private_capital(supabase_client=client)

Components:
    - PrivateCapitalTracker: Main tracker class
    - FundingEntry: Data class for funding announcements
    - IntensityResult: Calculation results
    - update_private_capital: Workflow function
    - get_bubble_index_interpretation: Cross-reference with Bubble Index
"""

from .private_capital import (
    PrivateCapitalTracker,
    FundingEntry,
    IntensityResult,
    PrivateCapitalConfig,
    SUPABASE_SCHEMA
)

from .workflow import (
    update_private_capital,
    get_bubble_index_interpretation,
    generate_private_capital_summary,
    run_monday_private_capital
)

from .sheets_writer import (
    write_private_capital_to_sheets,
    update_dashboard_private_capital
)

from .google_alerts_adapter import (
    GoogleAlertsAdapter,
    GOOGLE_ALERT_FEEDS,
    ALERT_QUERIES,
    collect_all_sources
)

__all__ = [
    'PrivateCapitalTracker',
    'FundingEntry',
    'IntensityResult',
    'PrivateCapitalConfig',
    'SUPABASE_SCHEMA',
    'update_private_capital',
    'get_bubble_index_interpretation',
    'generate_private_capital_summary',
    'run_monday_private_capital',
    'write_private_capital_to_sheets',
    'update_dashboard_private_capital',
    'GoogleAlertsAdapter',
    'GOOGLE_ALERT_FEEDS',
    'ALERT_QUERIES',
    'collect_all_sources'
]

__version__ = '1.0.0'