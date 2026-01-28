"""
PE Funding Parser
=================

Parses Google Alert entries into structured PE funding data.
Extracts company, amount, stage, sectors (emergent from NLP).

Author: ARGUS-1 Development Team
Created: January 2026
"""

import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# SECTOR KEYWORDS (emergent from data, not filtering)
# =============================================================================

SECTOR_KEYWORDS = {
    'ai_ml': ['artificial intelligence', 'machine learning', 'ai', 'ml', 'deep learning', 
              'neural network', 'llm', 'generative ai', 'gpt', 'foundation model'],
    'fintech': ['fintech', 'payments', 'banking', 'neobank', 'lending', 'insurtech',
                'crypto', 'blockchain', 'defi', 'trading'],
    'healthtech': ['healthtech', 'biotech', 'medtech', 'healthcare', 'drug discovery',
                   'clinical', 'pharma', 'therapeutics', 'diagnostics'],
    'climate': ['climate', 'cleantech', 'renewable', 'solar', 'wind', 'battery',
                'ev', 'electric vehicle', 'carbon', 'sustainability', 'green'],
    'enterprise': ['saas', 'enterprise', 'b2b', 'software', 'cloud', 'devops',
                   'cybersecurity', 'security', 'data', 'analytics'],
    'consumer': ['consumer', 'e-commerce', 'retail', 'marketplace', 'd2c',
                 'social', 'gaming', 'entertainment', 'media'],
    'infrastructure': ['infrastructure', 'semiconductor', 'chips', 'datacenter',
                       'networking', 'hardware', 'compute', 'gpu'],
    'defense': ['defense', 'aerospace', 'military', 'govtech', 'space'],
    'robotics': ['robotics', 'automation', 'manufacturing', 'industrial'],
}


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_amount(text: str) -> Optional[float]:
    """
    Extract funding amount from text.
    Returns amount in millions USD.
    """
    text_lower = text.lower()
    
    # Pattern: $X billion
    billion_match = re.search(r'\$(\d+(?:\.\d+)?)\s*billion', text_lower)
    if billion_match:
        return float(billion_match.group(1)) * 1000
    
    # Pattern: $X million or $Xm or $X mn
    million_match = re.search(r'\$(\d+(?:\.\d+)?)\s*(?:million|m\b|mn)', text_lower)
    if million_match:
        return float(million_match.group(1))
    
    # Pattern: $XXX million (with commas)
    comma_match = re.search(r'\$(\d{1,3}(?:,\d{3})*)\s*(?:million)?', text_lower)
    if comma_match:
        amount = float(comma_match.group(1).replace(',', ''))
        if amount >= 1000:  # Likely in thousands, convert to millions
            return amount / 1000
        return amount
    
    # Pattern: raised XXX million
    raised_match = re.search(r'raised\s+(\d+(?:\.\d+)?)\s*(?:million|m\b)', text_lower)
    if raised_match:
        return float(raised_match.group(1))
    
    return None


def extract_company(text: str) -> str:
    """
    Extract company name from headline.
    """
    # Pattern: "Company raises $X" or "Company closes $X"
    match = re.match(r'^([A-Z][A-Za-z0-9\s&\-\.]+?)(?:\s+raises|\s+closes|\s+secures|\s+lands|\s+gets)', text)
    if match:
        return match.group(1).strip()
    
    # Pattern: "Company, a startup" or "Company, the"
    match = re.match(r'^([A-Z][A-Za-z0-9\s&\-\.]+?),\s+(?:a|the|an)', text)
    if match:
        return match.group(1).strip()
    
    # Fallback: first few capitalized words
    words = text.split()
    company_words = []
    for word in words[:5]:
        if word and word[0].isupper() and word.lower() not in ['the', 'a', 'an', 'in', 'for', 'and']:
            company_words.append(word)
        elif company_words:
            break
    
    return ' '.join(company_words) if company_words else 'Unknown'


def extract_stage(text: str, alert_source: str) -> str:
    """
    Extract funding stage from text and alert source.
    """
    text_lower = text.lower()
    
    # Direct stage detection
    if 'series a' in text_lower:
        return 'Series_A'
    if 'series b' in text_lower:
        return 'Series_B'
    if 'series c' in text_lower:
        return 'Series_C'
    if 'series d' in text_lower:
        return 'Series_D'
    if 'series e' in text_lower or 'series f' in text_lower:
        return 'Series_E+'
    if 'growth round' in text_lower or 'growth equity' in text_lower:
        return 'Growth'
    if 'seed' in text_lower or 'pre-seed' in text_lower:
        return 'Seed'
    if 'ipo' in text_lower or 's-1' in text_lower or 'going public' in text_lower:
        return 'IPO'
    if 'venture fund' in text_lower or 'fund launch' in text_lower:
        return 'Fund_Launch'
    
    # Infer from alert source
    source_lower = alert_source.lower() if alert_source else ''
    if 'series_a' in source_lower:
        return 'Series_A'
    if 'series_b' in source_lower:
        return 'Series_B'
    if 'late_stage' in source_lower:
        return 'Late_Stage'
    if 'mega_rounds' in source_lower:
        return 'Mega_Round'
    if 'ipo_pipeline' in source_lower:
        return 'IPO'
    if 'vc_fund_launches' in source_lower or 'fund_launches' in source_lower:
        return 'Fund_Launch'
    
    return 'Unknown'


def extract_sectors(text: str) -> List[str]:
    """
    Extract sectors from text using keyword matching.
    Sectors emerge from the data, not predefined filtering.
    """
    text_lower = text.lower()
    detected = []
    
    for sector, keywords in SECTOR_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                if sector not in detected:
                    detected.append(sector)
                break
    
    return detected if detected else ['general']


def extract_keywords(text: str) -> List[str]:
    """
    Extract notable keywords/terms from text.
    """
    text_lower = text.lower()
    keywords = []
    
    # Tech buzzwords
    buzzwords = ['platform', 'saas', 'api', 'automation', 'analytics', 
                 'marketplace', 'network', 'infrastructure', 'cloud',
                 'mobile', 'enterprise', 'consumer', 'b2b', 'b2c']
    
    for word in buzzwords:
        if word in text_lower and word not in keywords:
            keywords.append(word)
    
    return keywords[:5]  # Limit to top 5


def extract_investors(text: str) -> List[str]:
    """
    Extract investor names from text.
    """
    # Common VC firm patterns
    vc_patterns = [
        r'led by ([A-Z][A-Za-z\s&]+?)(?:,|\.|and|with)',
        r'from ([A-Z][A-Za-z\s&]+?)(?:,|\.|and)',
        r'([A-Z][A-Za-z\s&]+?) led the',
        r'([A-Z][A-Za-z\s&]+?) participated',
    ]
    
    investors = []
    for pattern in vc_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            investor = match.strip()
            if len(investor) > 2 and investor not in investors:
                investors.append(investor)
    
    return investors[:5]  # Limit to top 5


# =============================================================================
# MAIN PARSER
# =============================================================================

def parse_pe_alert(alert, min_amount_m: float = 5.0) -> Optional[Dict[str, Any]]:
    """
    Parse a Google Alert entry into a PE funding entry dict.
    
    Args:
        alert: GoogleAlertEntry object with title, link, published, source
        min_amount_m: Minimum amount in millions to include
        
    Returns:
        Dict with funding entry data, or None if not valid
    """
    title = alert.title
    source = alert.source
    
    # Extract amount
    amount_m = extract_amount(title)
    
    # For IPO and Fund Launch alerts, amount may be missing
    if source not in ['ipo_pipeline', 'vc_fund_launches']:
        if not amount_m or amount_m < min_amount_m:
            return None
    
    entry = {
        'date': alert.published.strftime('%Y-%m-%d'),
        'company': extract_company(title),
        'amount_m': amount_m or 0,
        'stage': extract_stage(title, source),
        'raw_text': title,
        'source': f'PE_Alert_{source}',
        'url': alert.link,
        'sectors': extract_sectors(title),
        'keywords': extract_keywords(title),
        'investors': extract_investors(title),
    }
    
    return entry