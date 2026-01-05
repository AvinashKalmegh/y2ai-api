"""
Historical Fingerprint Matching

Compares current market conditions to historical episodes
using weighted Euclidean distance across 5 dimensions.
Matches Google Sheets HistoricalFingerprints.gs methodology.
"""

import logging
import math
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EpisodeFingerprint:
    """Fingerprint values for a market episode."""
    amri: float          # 0-100
    bubble: float        # 0-100
    contagion: float     # 0-100
    correlations: float  # 0-1
    clusters: int        # fewer = more stress


@dataclass
class HistoricalEpisode:
    """A known historical market episode."""
    name: str
    date: str
    fingerprint: EpisodeFingerprint
    pattern: str
    what_happened: str
    key_feature: str


# Historical episodes - matches Google Sheets HISTORICAL_EPISODES
HISTORICAL_EPISODES = [
    HistoricalEpisode(
        name="COVID Crash",
        date="Mar 2020",
        fingerprint=EpisodeFingerprint(amri=85, bubble=45, contagion=95, correlations=0.90, clusters=3),
        pattern="Sudden spike, V-recovery",
        what_happened="Markets dropped 34% in 23 days, then recovered 68% in 3 months",
        key_feature="Extreme correlation spike with liquidity crisis"
    ),
    HistoricalEpisode(
        name="2022 Tech Correction",
        date="Jan-Oct 2022",
        fingerprint=EpisodeFingerprint(amri=65, bubble=72, contagion=70, correlations=0.65, clusters=5),
        pattern="Grinding decline",
        what_happened="NASDAQ fell 33% over 10 months, slow recovery",
        key_feature="Elevated bubble index with rising rates pressure"
    ),
    HistoricalEpisode(
        name="SVB Banking Crisis",
        date="Mar 2023",
        fingerprint=EpisodeFingerprint(amri=55, bubble=48, contagion=80, correlations=0.55, clusters=4),
        pattern="Sector contagion",
        what_happened="Regional banks collapsed, contained within 2 weeks",
        key_feature="High contagion but low bubble - systemic not speculative"
    ),
    HistoricalEpisode(
        name="Dot-com Peak",
        date="Mar 2000",
        fingerprint=EpisodeFingerprint(amri=80, bubble=94, contagion=85, correlations=0.75, clusters=3),
        pattern="Bubble burst",
        what_happened="NASDAQ fell 78% over 2.5 years",
        key_feature="Extreme bubble index with euphoric sentiment"
    ),
    HistoricalEpisode(
        name="Aug 2024 Yen Unwind",
        date="Aug 2024",
        fingerprint=EpisodeFingerprint(amri=60, bubble=55, contagion=75, correlations=0.70, clusters=4),
        pattern="Flash crash, quick recovery",
        what_happened="3-day 6% drop, recovered in 2 weeks",
        key_feature="External shock (yen carry trade) with healthy fundamentals"
    ),
    HistoricalEpisode(
        name="Normal/Baseline",
        date="Typical",
        fingerprint=EpisodeFingerprint(amri=25, bubble=35, contagion=30, correlations=0.35, clusters=12),
        pattern="Healthy market",
        what_happened="Normal trading conditions",
        key_feature="Low stress across all metrics"
    )
]


@dataclass
class FingerprintMatch:
    """Result of fingerprint matching"""
    episode: str
    date: str
    distance: float           # 0-100, lower = closer match
    match_quality: str        # STRONG, MODERATE, WEAK, UNIQUE
    pattern: str
    what_happened: str
    key_feature: str
    key_differences: List[str]
    all_matches: List[Dict]
    display: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class FingerprintMatcher:
    """
    Match current conditions to historical episodes using
    weighted Euclidean distance (Google Sheets methodology).
    """
    
    WEIGHTS = {
        'amri': 1.0,
        'bubble': 1.0,
        'contagion': 1.2,
        'correlations': 0.8,
        'clusters': 0.6
    }
    
    QUALITY_THRESHOLDS = {
        'STRONG': 15,
        'MODERATE': 30,
        'WEAK': 45
    }
    
    def __init__(self, episodes: Optional[List[HistoricalEpisode]] = None):
        self.episodes = episodes if episodes else HISTORICAL_EPISODES
    
    def calculate_distance(self, current: Dict[str, float], historical: EpisodeFingerprint) -> float:
        """Weighted Euclidean distance, 0-100 scale."""
        sum_squares = 0.0
        total_weight = 0.0
        
        # AMRI
        amri_diff = (current['amri'] - historical.amri) / 100
        sum_squares += self.WEIGHTS['amri'] * amri_diff ** 2
        total_weight += self.WEIGHTS['amri']
        
        # Bubble
        bubble_diff = (current['bubble'] - historical.bubble) / 100
        sum_squares += self.WEIGHTS['bubble'] * bubble_diff ** 2
        total_weight += self.WEIGHTS['bubble']
        
        # Contagion
        contagion_diff = (current['contagion'] - historical.contagion) / 100
        sum_squares += self.WEIGHTS['contagion'] * contagion_diff ** 2
        total_weight += self.WEIGHTS['contagion']
        
        # Correlations
        corr_diff = current['correlations'] - historical.correlations
        sum_squares += self.WEIGHTS['correlations'] * corr_diff ** 2
        total_weight += self.WEIGHTS['correlations']
        
        # Clusters (invert: fewer = more stress)
        current_stress = max(0, (15 - current['clusters']) / 12 * 100)
        historical_stress = max(0, (15 - historical.clusters) / 12 * 100)
        cluster_diff = (current_stress - historical_stress) / 100
        sum_squares += self.WEIGHTS['clusters'] * cluster_diff ** 2
        total_weight += self.WEIGHTS['clusters']
        
        return round(math.sqrt(sum_squares / total_weight) * 100, 1)
    
    def get_quality(self, distance: float) -> str:
        if distance < self.QUALITY_THRESHOLDS['STRONG']:
            return 'STRONG'
        elif distance < self.QUALITY_THRESHOLDS['MODERATE']:
            return 'MODERATE'
        elif distance < self.QUALITY_THRESHOLDS['WEAK']:
            return 'WEAK'
        return 'UNIQUE'
    
    def find_differences(self, current: Dict[str, float], historical: EpisodeFingerprint) -> List[str]:
        """Key differences between current and historical."""
        diffs = []
        
        if abs(current['amri'] - historical.amri) > 15:
            dir = "higher" if current['amri'] > historical.amri else "lower"
            diffs.append(f"AMRI {dir} ({int(current['amri'])} vs {historical.amri})")
        
        if abs(current['bubble'] - historical.bubble) > 15:
            dir = "higher" if current['bubble'] > historical.bubble else "lower"
            diffs.append(f"Bubble {dir} ({int(current['bubble'])} vs {historical.bubble})")
        
        if abs(current['contagion'] - historical.contagion) > 15:
            dir = "higher" if current['contagion'] > historical.contagion else "lower"
            diffs.append(f"Contagion {dir} ({int(current['contagion'])} vs {historical.contagion})")
        
        if abs(current['correlations'] - historical.correlations) > 0.15:
            dir = "higher" if current['correlations'] > historical.correlations else "lower"
            diffs.append(f"Correlations {dir} ({current['correlations']:.0%} vs {historical.correlations:.0%})")
        
        if abs(current['clusters'] - historical.clusters) > 3:
            dir = "more" if current['clusters'] > historical.clusters else "fewer"
            diffs.append(f"Clusters {dir} ({int(current['clusters'])} vs {historical.clusters})")
        
        return diffs if diffs else ["Minor differences across all metrics"]
    
    def calculate(
        self,
        amri: float = 38.0,
        bubble: float = 57.0,
        contagion: float = 50.0,
        correlations: float = 0.42,
        clusters: int = 7
    ) -> FingerprintMatch:
        """Find best matching historical episode."""
        
        current = {
            'amri': amri,
            'bubble': bubble,
            'contagion': contagion,
            'correlations': correlations,
            'clusters': clusters
        }
        
        matches = []
        for ep in self.episodes:
            dist = self.calculate_distance(current, ep.fingerprint)
            matches.append({
                'name': ep.name,
                'date': ep.date,
                'distance': dist,
                'pattern': ep.pattern
            })
        
        matches.sort(key=lambda x: x['distance'])
        best = matches[0]
        
        best_ep = next(e for e in self.episodes if e.name == best['name'])
        quality = self.get_quality(best['distance'])
        diffs = self.find_differences(current, best_ep.fingerprint)
        
        # Display format
        display = f"{best['name']} ({best['distance']})"
        if quality == 'STRONG':
            display = f"⚡ {display}"
        elif quality == 'MODERATE':
            display = f"📊 {display}"
        
        return FingerprintMatch(
            episode=best['name'],
            date=best['date'],
            distance=best['distance'],
            match_quality=quality,
            pattern=best['pattern'],
            what_happened=best_ep.what_happened,
            key_feature=best_ep.key_feature,
            key_differences=diffs,
            all_matches=matches[:3],
            display=display
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    matcher = FingerprintMatcher()
    
    # Test with Dec 2025 values (matches Google Sheets test)
    result = matcher.calculate(
        amri=33.3,
        bubble=57,
        contagion=50.0,
        correlations=0.42,
        clusters=7
    )
    
    print(f"\n{'='*60}")
    print("HISTORICAL FINGERPRINT")
    print(f"{'='*60}")
    print(f"Nearest Match: {result.episode} ({result.date})")
    print(f"Distance: {result.distance} ({result.match_quality})")
    print(f"Pattern: {result.pattern}")
    print(f"What Happened: {result.what_happened}")
    print(f"Key Differences: {'; '.join(result.key_differences)}")
    print(f"\nTop 3 Matches:")
    for i, m in enumerate(result.all_matches, 1):
        print(f"  {i}. {m['name']} (distance: {m['distance']})")
    print(f"\nDisplay: {result.display}")