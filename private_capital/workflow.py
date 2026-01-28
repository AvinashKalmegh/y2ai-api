"""
Private Capital Workflow Integration
====================================

Integrates Private Capital Tracker into ARGUS-1 weekly workflow.
Runs Monday mornings as part of the weekly update cycle.

Author: ARGUS-1 Development Team
Created: January 2026
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def update_private_capital(supabase_client=None, write_to_sheets: bool = True):
    """
    Weekly private capital update - runs Monday.
    
    Args:
        supabase_client: Supabase client for database operations
        write_to_sheets: Whether to write to Google Sheets (default True)
        
    Returns:
        Dict with update results and intensity score
    """
    from .private_capital import PrivateCapitalTracker
    
    logger.info("=" * 60)
    logger.info("PRIVATE CAPITAL UPDATE (Monday Workflow)")
    logger.info("=" * 60)
    
    tracker = PrivateCapitalTracker(supabase_client=supabase_client)
    result = tracker.update_dial()
    
    intensity = result['intensity']
    
    # Log summary
    logger.info(f"Private Capital Update Complete:")
    logger.info(f"  New entries: {result['new_entries']}")
    logger.info(f"  Intensity: {intensity.score} ({intensity.regime})")
    logger.info(f"  30D Volume: ${intensity.vol_30d_m/1000:.1f}B")
    logger.info(f"  90D Volume: ${intensity.vol_90d_m/1000:.1f}B")
    logger.info(f"  Megarounds: {intensity.megarounds_30d}")
    logger.info(f"  Categories: {intensity.categories}")
    
    # Write to Google Sheets
    if write_to_sheets:
        try:
            from .sheets_writer import write_private_capital_to_sheets
            write_private_capital_to_sheets(result)
        except Exception as e:
            logger.warning(f"Could not write to Google Sheets: {e}")
    
    return result


def get_bubble_index_interpretation(intensity_score: int, intensity_regime: str, bubble_index: int) -> dict:
    """
    Interpret Private Capital signal in context of Bubble Index.
    
    This cross-reference provides cycle positioning:
    - High PC + Low Bubble = Early cycle infrastructure buildout
    - High PC + High Bubble = Late-cycle exuberance
    - Low PC + High Bubble = Distribution (institutions exiting)
    - Low PC + Low Bubble = Capital freeze
    
    Args:
        intensity_score: Private Capital Intensity (0-100)
        intensity_regime: PC regime (FROZEN/WEAK/NORMAL/STRONG/SURGE)
        bubble_index: ARGUS-1 Bubble Index (0-100)
        
    Returns:
        Dict with interpretation and recommended action
    """
    interpretation = {
        'pc_score': intensity_score,
        'pc_regime': intensity_regime,
        'bubble_index': bubble_index,
        'signal': '',
        'interpretation': '',
        'infra_bias': 0,  # -1 to +1 adjustment for Infrastructure pillar
    }
    
    # Classify bubble regime
    if bubble_index >= 70:
        bubble_regime = 'HIGH'
    elif bubble_index >= 50:
        bubble_regime = 'ELEVATED'
    else:
        bubble_regime = 'LOW'
    
    # Cross-reference matrix
    if intensity_regime in ['SURGE', 'STRONG'] and bubble_regime == 'LOW':
        interpretation['signal'] = 'EARLY_CYCLE'
        interpretation['interpretation'] = (
            'Strong private capital formation with low bubble risk. '
            'Classic infrastructure buildout phase. Smart money is deploying.'
        )
        interpretation['infra_bias'] = +0.5
        
    elif intensity_regime in ['SURGE', 'STRONG'] and bubble_regime == 'HIGH':
        interpretation['signal'] = 'LATE_CYCLE'
        interpretation['interpretation'] = (
            'High private capital AND high bubble risk. '
            'Late-cycle exuberance. Peak bubble risk - watch for sponsors to exit.'
        )
        interpretation['infra_bias'] = -0.25
        
    elif intensity_regime in ['SURGE', 'STRONG'] and bubble_regime == 'ELEVATED':
        interpretation['signal'] = 'CONFIRMATION'
        interpretation['interpretation'] = (
            'Strong private capital aligns with elevated market. '
            'Institutions confirm public market thesis. Continue allocation.'
        )
        interpretation['infra_bias'] = +0.25
        
    elif intensity_regime in ['WEAK', 'FROZEN'] and bubble_regime == 'HIGH':
        interpretation['signal'] = 'DISTRIBUTION'
        interpretation['interpretation'] = (
            'Private capital retreating while public bubble persists. '
            'Institutions are exiting. High risk of correction.'
        )
        interpretation['infra_bias'] = -0.5
        
    elif intensity_regime in ['WEAK', 'FROZEN'] and bubble_regime in ['LOW', 'ELEVATED']:
        interpretation['signal'] = 'CAPITAL_FREEZE'
        interpretation['interpretation'] = (
            'Private capital markets are cold. '
            'May indicate credit stress or macro uncertainty. Defensive posture.'
        )
        interpretation['infra_bias'] = -0.25
        
    else:  # NORMAL regime
        interpretation['signal'] = 'NEUTRAL'
        interpretation['interpretation'] = (
            'Normal private capital activity. '
            'No strong signal from VC/PE flows. Follow other indicators.'
        )
        interpretation['infra_bias'] = 0
    
    return interpretation


def generate_private_capital_summary(result: dict) -> str:
    """
    Generate human-readable summary for LinkedIn/newsletter.
    
    Args:
        result: Output from update_private_capital()
        
    Returns:
        Formatted summary string
    """
    intensity = result['intensity']
    
    lines = [
        f"Private Capital Intensity: {intensity.score}/100 ({intensity.regime})",
        f"",
        f"30-Day Volume: ${intensity.vol_30d_m/1000:.1f}B",
        f"Megarounds (>$500M): {intensity.megarounds_30d}",
        f"Active Categories: {', '.join(intensity.categories) if intensity.categories else 'None'}",
    ]
    
    # Add regime-specific commentary
    if intensity.regime == 'SURGE':
        lines.append("")
        lines.append("Exceptional capital formation. VCs are writing checks aggressively.")
    elif intensity.regime == 'STRONG':
        lines.append("")
        lines.append("Above-average VC activity. Private markets remain confident.")
    elif intensity.regime == 'WEAK':
        lines.append("")
        lines.append("Below-average funding. Private investors showing caution.")
    elif intensity.regime == 'FROZEN':
        lines.append("")
        lines.append("Capital markets are cold. Watch for credit stress signals.")
    
    return '\n'.join(lines)


def run_monday_private_capital(supabase_client=None, bubble_index: int = None):
    """
    Complete Monday workflow for Private Capital.
    
    This is called by the orchestrator on Monday mornings.
    
    Args:
        supabase_client: Supabase client
        bubble_index: Current Bubble Index value (if known)
        
    Returns:
        Dict with full results including interpretation
    """
    # Run update (without GS write yet - we'll add interpretation first)
    result = update_private_capital(supabase_client=supabase_client, write_to_sheets=False)
    
    # Get interpretation if Bubble Index provided
    if bubble_index is not None:
        interpretation = get_bubble_index_interpretation(
            result['intensity'].score,
            result['intensity'].regime,
            bubble_index
        )
        result['interpretation'] = interpretation
        
        logger.info(f"\n=== BUBBLE INDEX CROSS-REFERENCE ===")
        logger.info(f"PC: {interpretation['pc_score']} ({interpretation['pc_regime']}) | Bubble: {bubble_index}")
        logger.info(f"Signal: {interpretation['signal']}")
        logger.info(f"Infra Bias: {interpretation['infra_bias']:+.2f}")
        logger.info(f"{interpretation['interpretation']}")
    
    # Now write to Google Sheets with interpretation included
    try:
        from .sheets_writer import write_private_capital_to_sheets
        write_private_capital_to_sheets(result)
    except Exception as e:
        logger.warning(f"Could not write to Google Sheets: {e}")
    
    return result


if __name__ == '__main__':
    """Test workflow integration."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Private Capital Workflow')
    parser.add_argument('--test', action='store_true', help='Test update cycle')
    parser.add_argument('--interpret', action='store_true', help='Test interpretation')
    
    args = parser.parse_args()
    
    if args.test:
        print("\n=== Testing Private Capital Workflow ===\n")
        result = update_private_capital(supabase_client=None)
        print("\n=== Summary ===\n")
        print(generate_private_capital_summary(result))
    
    elif args.interpret:
        print("\n=== Testing Interpretation Matrix ===\n")
        
        scenarios = [
            (75, 'STRONG', 30, 'Early Cycle'),
            (80, 'SURGE', 75, 'Late Cycle'),
            (65, 'STRONG', 55, 'Confirmation'),
            (20, 'WEAK', 70, 'Distribution'),
            (15, 'FROZEN', 40, 'Capital Freeze'),
            (45, 'NORMAL', 50, 'Neutral'),
        ]
        
        for pc_score, pc_regime, bubble_idx, expected in scenarios:
            result = get_bubble_index_interpretation(pc_score, pc_regime, bubble_idx)
            print(f"PC: {pc_score} ({pc_regime}) | Bubble: {bubble_idx}")
            print(f"  Signal: {result['signal']} (expected: {expected})")
            print(f"  Infra Bias: {result['infra_bias']:+.2f}")
            print()
    
    else:
        parser.print_help()
