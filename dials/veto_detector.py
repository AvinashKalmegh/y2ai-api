"""
VETO DETECTOR V2
================
Keyword-based veto detection with severity tiers.

V2 Changes (January 2026):
- Severity tiers: ENACTED > PROPOSED_FEDERAL > PROPOSED_LOCAL
- Word boundary matching (prevents "ban" matching "Bank")
- Positive sentiment skip list
- Context requirement (must be AI/tech related)
- WARNING state for high-severity but below veto threshold

Author: ARGUS-1 Development Team
Created: January 2026
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Veto keywords by type
VETO_KEYWORDS = {
    'regulatory': [
        "ban", "banned", "banning", "moratorium", "shutdown", "shut down",
        "halted", "halt", "investigation", "inquiry", "probe", "antitrust",
        "blocked", "blocking", "prohibited", "prohibition", "suspended",
        "sanctions", "export controls", "blacklist", "entity list"
    ],
    'grid': [
        "power cuts", "power outage", "grid overload", "grid failure",
        "connection refused", "capacity shortage",
        "curtailment", "blackout", "brownout", "utility denied", "permit denied"
    ],
    'financing': [
        "credit freeze", "no bid", "failed bond sale", "withdrawn offering",
        "covenant breach", "downgrade", "liquidity crisis", "margin call",
        "redemption halt", "default", "restructuring"
    ],
    'macro': [
        "taiwan strait", "taiwan invasion", "taiwan blockade",
        "semiconductor embargo", "rare earth embargo", "tech cold war",
        "chips act revoked", "systemic risk", "contagion", "flash crash"
    ]
}

# Severity modifiers for veto tiers
VETO_SEVERITY = {
    'ENACTED': 1.0,           # Full veto - actual policy in effect
    'PROPOSED_FEDERAL': 0.6,  # Warning - federal proposal, not enacted
    'PROPOSED_LOCAL': 0.3,    # Monitor - state/local proposal
    'NOISE': 0.0              # Ignore - media speculation
}

# Proposal indicator keywords (reduce severity)
PROPOSAL_KEYWORDS = [
    "seeks", "proposes", "considers", "considering", "may", "might", 
    "could", "would", "calls for", "urges", "requests", "demands",
    "coalition", "advocacy", "petition", "lobbying", "debate",
    "discusses", "weighs", "mulls", "eyes", "floats", "suggests"
]

# Local/state indicators (reduce severity further)
LOCAL_KEYWORDS = [
    "state", "county", "city", "local", "municipal", "town", "township",
    "oklahoma", "wisconsin", "janesville", "oregon", "virginia", "texas",
    "california", "arizona", "georgia", "ohio", "indiana", "tennessee",
    "north carolina", "south carolina", "michigan", "pennsylvania",
    "senator proposes", "assemblyman", "councilman", "mayor"
]

# Enacted indicators (full severity)
ENACTED_KEYWORDS = [
    "passed", "enacted", "signed into law", "effective immediately",
    "implemented", "now in effect", "takes effect", "approved",
    "ratified", "executive order", "officially", "law goes into effect",
    "regulation finalized", "rule finalized", "formally adopted"
]

# Positive indicators - skip veto detection for these
POSITIVE_INDICATORS = [
    'boost', 'wins', 'success', 'growth', 'climbs', 'emerging as winner', 
    'approved for', 'cleared for', 'green light', 'permits granted',
    'approves', 'approved', 'lifts', 'lifted', 'eases', 'eased',
    'relaxes', 'relaxed', 'exemption', 'exempted', 'license granted',
    'us approves', 'trump approves', 'biden approves', 'commerce approves',
    'exports to china', 'cleared for china', 'thaw', 'warming'
]

# Context words - must be AI/tech related
CONTEXT_WORDS = [
    'ai', 'data center', 'chip', 'semiconductor', 'cloud', 'nvidia', 
    'tech', 'gpu', 'datacenter', 'artificial intelligence', 'infrastructure',
    'compute', 'server', 'hyperscaler', 'power', 'energy', 'grid'
]

# Thresholds
VETO_THRESHOLD = 0.6      # Full veto trigger (effective_impact >= 0.6)
WARNING_THRESHOLD = 0.4   # Warning trigger (effective_impact >= 0.4)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VetoArticle:
    """Article that triggered a veto."""
    title: str
    keyword: str
    impact: float
    effective_impact: float
    severity_reason: str


@dataclass
class VetoTypeDetail:
    """Details for a single veto type."""
    active: bool = False
    severity: float = 0.0
    status: str = "CLEAR"  # VETO, WARNING, CLEAR
    articles: List[VetoArticle] = field(default_factory=list)


@dataclass
class VetoResult:
    """Complete veto detection result."""
    # Legacy format (for backward compatibility)
    vetos: Dict[str, bool] = field(default_factory=dict)
    # New detailed format
    veto_details: Dict[str, VetoTypeDetail] = field(default_factory=dict)
    # Summary
    has_veto: bool = False
    has_warning: bool = False
    veto_types: List[str] = field(default_factory=list)
    warning_types: List[str] = field(default_factory=list)
    max_severity: float = 0.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def escape_regex(string: str) -> str:
    """Escape special regex characters."""
    return re.escape(string)


def parse_json_field(value: Any) -> List[str]:
    """Parse a field that might be JSON string or list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except:
            return []
    return []


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

def check_veto_conditions(articles: List[Dict[str, Any]]) -> VetoResult:
    """
    Check for veto conditions in articles.
    V2: With severity tiers (ENACTED > PROPOSED_FEDERAL > PROPOSED_LOCAL)
    
    Args:
        articles: List of article dicts with title, key_quotes, extracted_facts, impact_score
        
    Returns:
        VetoResult with vetos, veto_details, and summary flags
    """
    veto_details = {
        'regulatory': VetoTypeDetail(),
        'grid': VetoTypeDetail(),
        'financing': VetoTypeDetail(),
        'macro': VetoTypeDetail()
    }
    
    used_article_ids = set()
    
    for article in articles:
        # Get article identifier
        article_id = article.get('id') or article.get('article_hash') or article.get('title', '')
        if article_id in used_article_ids:
            continue
        
        # Build text to scan
        title = article.get('title', '')
        key_quotes = parse_json_field(article.get('key_quotes', []))
        extracted_facts = parse_json_field(article.get('extracted_facts', []))
        
        text_to_scan = ' '.join([
            title,
            ' '.join(key_quotes),
            ' '.join(extracted_facts)
        ]).lower()
        
        # Check for context (must be AI/tech related)
        has_context = any(ctx in text_to_scan for ctx in CONTEXT_WORDS)
        if not has_context:
            continue
        
        # Skip if positive sentiment
        is_positive = any(pos in text_to_scan for pos in POSITIVE_INDICATORS)
        if is_positive:
            continue
        
        # Additional check: export approval context
        if 'export' in text_to_scan:
            if any(word in text_to_scan for word in ['approv', 'clear', 'lift', 'ease', 'strategic', 'license']):
                continue
        
        # Determine severity modifier
        is_proposal = any(kw in text_to_scan for kw in PROPOSAL_KEYWORDS)
        is_local = any(kw in text_to_scan for kw in LOCAL_KEYWORDS)
        is_enacted = any(kw in text_to_scan for kw in ENACTED_KEYWORDS)
        
        if is_enacted:
            severity_mod = VETO_SEVERITY['ENACTED']
            severity_reason = 'enacted'
        elif is_proposal and is_local:
            severity_mod = VETO_SEVERITY['PROPOSED_LOCAL']
            severity_reason = 'local_proposal'
        elif is_proposal:
            severity_mod = VETO_SEVERITY['PROPOSED_FEDERAL']
            severity_reason = 'federal_proposal'
        elif is_local:
            severity_mod = VETO_SEVERITY['PROPOSED_LOCAL']
            severity_reason = 'local'
        else:
            severity_mod = VETO_SEVERITY['ENACTED']
            severity_reason = 'enacted'
        
        # Calculate effective impact
        impact_score = float(article.get('impact_score', 0.5))
        effective_impact = impact_score * severity_mod
        
        # Skip low-severity items
        if effective_impact < 0.3:
            continue
        
        # Find veto type (priority: macro > financing > grid > regulatory)
        matched_veto = None
        matched_keyword = None
        
        for veto_type in ['macro', 'financing', 'grid', 'regulatory']:
            for keyword in VETO_KEYWORDS[veto_type]:
                # Word boundary matching to avoid "ban" in "Bank"
                pattern = r'\b' + escape_regex(keyword.lower()) + r'\b'
                if re.search(pattern, text_to_scan, re.IGNORECASE):
                    matched_veto = veto_type
                    matched_keyword = keyword
                    break
            if matched_veto:
                break
        
        # Record match
        if matched_veto:
            used_article_ids.add(article_id)
            veto_data = veto_details[matched_veto]
            
            # Update status based on effective impact
            if effective_impact >= VETO_THRESHOLD:
                veto_data.active = True
                veto_data.status = 'VETO'
            elif effective_impact >= WARNING_THRESHOLD and veto_data.status != 'VETO':
                veto_data.status = 'WARNING'
            
            # Track max severity
            if effective_impact > veto_data.severity:
                veto_data.severity = effective_impact
            
            veto_data.articles.append(VetoArticle(
                title=title,
                keyword=matched_keyword,
                impact=impact_score,
                effective_impact=effective_impact,
                severity_reason=severity_reason
            ))
    
    # Build result
    vetos = {k: v.active for k, v in veto_details.items()}
    veto_types = [k for k, v in veto_details.items() if v.active]
    warning_types = [k for k, v in veto_details.items() if v.status == 'WARNING' and not v.active]
    max_severity = max((v.severity for v in veto_details.values()), default=0.0)
    
    return VetoResult(
        vetos=vetos,
        veto_details=veto_details,
        has_veto=any(vetos.values()),
        has_warning=len(warning_types) > 0,
        veto_types=veto_types,
        warning_types=warning_types,
        max_severity=max_severity
    )


def calculate_veto_score(veto_result: VetoResult, topic_max_risk: float = 0.0) -> Dict[str, Any]:
    """
    Calculate final veto-adjusted score.
    
    Args:
        veto_result: Result from check_veto_conditions
        topic_max_risk: Maximum topic risk (0-100) from topic analysis
        
    Returns:
        Dict with score, regime, bubble_adjustment
    """
    # Active vetos - full VETO regime
    if veto_result.has_veto:
        score = 90 + (len(veto_result.veto_types) * 2.5)
        return {
            'score': min(score, 100),
            'regime': 'VETO',
            'veto_types': veto_result.veto_types,
            'warning_types': [],
            'bubble_adjustment': 25
        }
    
    # Warnings - elevated but not full veto
    if veto_result.has_warning:
        warning_score = 60 + (veto_result.max_severity * 30)  # 60-78 range
        return {
            'score': warning_score,
            'regime': 'High',
            'veto_types': [],
            'warning_types': veto_result.warning_types,
            'bubble_adjustment': 15
        }
    
    # No veto or warning - use topic risk
    if topic_max_risk >= 80:
        regime = 'Extreme'
        bubble_adjustment = 25
    elif topic_max_risk >= 60:
        regime = 'High'
        bubble_adjustment = 15
    elif topic_max_risk >= 40:
        regime = 'Medium'
        bubble_adjustment = 5
    else:
        regime = 'Low'
        bubble_adjustment = 0
    
    return {
        'score': topic_max_risk,
        'regime': regime,
        'veto_types': [],
        'warning_types': [],
        'bubble_adjustment': bubble_adjustment
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_veto_summary(veto_result: VetoResult) -> str:
    """Get human-readable summary of veto status."""
    if veto_result.has_veto:
        types = ', '.join(veto_result.veto_types)
        return f"VETO ACTIVE: {types}"
    elif veto_result.has_warning:
        types = ', '.join(veto_result.warning_types)
        return f"WARNING: {types}"
    else:
        return "CLEAR"


def veto_result_to_dict(veto_result: VetoResult) -> Dict[str, Any]:
    """Convert VetoResult to dictionary for JSON serialization."""
    return {
        'vetos': veto_result.vetos,
        'veto_details': {
            k: {
                'active': v.active,
                'severity': v.severity,
                'status': v.status,
                'article_count': len(v.articles),
                'articles': [
                    {
                        'title': a.title,
                        'keyword': a.keyword,
                        'impact': a.impact,
                        'effective_impact': a.effective_impact,
                        'severity_reason': a.severity_reason
                    }
                    for a in v.articles
                ]
            }
            for k, v in veto_result.veto_details.items()
        },
        'has_veto': veto_result.has_veto,
        'has_warning': veto_result.has_warning,
        'veto_types': veto_result.veto_types,
        'warning_types': veto_result.warning_types,
        'max_severity': veto_result.max_severity,
        'summary': get_veto_summary(veto_result)
    }


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    # Test with sample articles
    test_articles = [
        {
            'title': 'Coalition seeks nationwide moratorium on data centers',
            'key_quotes': ['Environmental groups demand halt to construction'],
            'extracted_facts': ['Proposed legislation would ban new facilities'],
            'impact_score': 0.9
        },
        {
            'title': 'NVIDIA H200 exports approved for strategic partners',
            'key_quotes': ['Commerce Department clears shipments'],
            'extracted_facts': ['Licenses granted for AI chip exports'],
            'impact_score': 0.8
        },
        {
            'title': 'Texas grid operator warns of capacity shortage',
            'key_quotes': ['ERCOT cites summer strain on infrastructure'],
            'extracted_facts': ['Grid overload concerns mounting'],
            'impact_score': 0.85
        }
    ]
    
    print("=" * 60)
    print("VETO DETECTOR V2 TEST")
    print("=" * 60)
    
    result = check_veto_conditions(test_articles)
    
    print(f"\nSummary: {get_veto_summary(result)}")
    print(f"Has Veto: {result.has_veto}")
    print(f"Has Warning: {result.has_warning}")
    print(f"Max Severity: {result.max_severity:.2f}")
    
    print("\nVeto Details:")
    for veto_type, detail in result.veto_details.items():
        if detail.articles:
            print(f"\n  {veto_type.upper()}:")
            print(f"    Status: {detail.status}")
            print(f"    Severity: {detail.severity:.2f}")
            for article in detail.articles:
                print(f"    - {article.title[:50]}...")
                print(f"      Keyword: {article.keyword}, Reason: {article.severity_reason}")
    
    # Test score calculation
    score_result = calculate_veto_score(result, topic_max_risk=45.0)
    print(f"\nScore: {score_result['score']:.1f}")
    print(f"Regime: {score_result['regime']}")
    print(f"Bubble Adjustment: +{score_result['bubble_adjustment']}")