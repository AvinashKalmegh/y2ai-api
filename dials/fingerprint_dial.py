"""
FINGERPRINT LIBRARY DIAL
========================
Compare current market state to historical stress episodes.

Historical Episodes:
- COVID Crash (Mar 2020): V-recovery, extreme correlation spike
- 2022 Tech Correction (Jan-Oct 2022): Grinding decline, rate pressure
- SVB Banking Crisis (Mar 2023): Sector contagion, contained
- Dot-com Peak (Mar 2000): Bubble burst, 78% decline
- Aug 2024 Yen Unwind: Flash crash, quick recovery
- Normal/Baseline: Healthy market conditions

Fingerprint Metrics:
- AMRI (0-100)
- Bubble Index (0-100)
- Contagion Score (0-100)
- Correlations (0-1)
- Cluster Count (3-15)

Match Quality:
- STRONG: Distance < 15
- MODERATE: Distance 15-30
- WEAK: Distance 30-50
- UNIQUE: Distance > 50
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# HISTORICAL EPISODES
# =============================================================================

HISTORICAL_EPISODES = [
    {
        "name": "COVID Crash",
        "date": "Mar 2020",
        "fingerprint": {"amri": 85, "bubble": 45, "contagion": 95, "correlations": 0.90, "clusters": 3},
        "pattern": "Sudden spike, V-recovery",
        "what_happened": "Markets dropped 34% in 23 days, then recovered 68% in 3 months",
        "key_feature": "Extreme correlation spike with liquidity crisis"
    },
    {
        "name": "2022 Tech Correction",
        "date": "Jan-Oct 2022",
        "fingerprint": {"amri": 65, "bubble": 72, "contagion": 70, "correlations": 0.65, "clusters": 5},
        "pattern": "Grinding decline",
        "what_happened": "NASDAQ fell 33% over 10 months, slow recovery",
        "key_feature": "Elevated bubble index with rising rates pressure"
    },
    {
        "name": "SVB Banking Crisis",
        "date": "Mar 2023",
        "fingerprint": {"amri": 55, "bubble": 48, "contagion": 80, "correlations": 0.55, "clusters": 4},
        "pattern": "Sector contagion",
        "what_happened": "Regional banks collapsed, contained within 2 weeks",
        "key_feature": "High contagion but low bubble - systemic not speculative"
    },
    {
        "name": "Dot-com Peak",
        "date": "Mar 2000",
        "fingerprint": {"amri": 80, "bubble": 94, "contagion": 85, "correlations": 0.75, "clusters": 3},
        "pattern": "Bubble burst",
        "what_happened": "NASDAQ fell 78% over 2.5 years",
        "key_feature": "Extreme bubble index with euphoric sentiment"
    },
    {
        "name": "Aug 2024 Yen Unwind",
        "date": "Aug 2024",
        "fingerprint": {"amri": 60, "bubble": 55, "contagion": 75, "correlations": 0.70, "clusters": 4},
        "pattern": "Flash crash, quick recovery",
        "what_happened": "3-day 6% drop, recovered in 2 weeks",
        "key_feature": "External shock (yen carry trade) with healthy fundamentals"
    },
    {
        "name": "Normal/Baseline",
        "date": "Typical",
        "fingerprint": {"amri": 25, "bubble": 35, "contagion": 30, "correlations": 0.35, "clusters": 12},
        "pattern": "Healthy market",
        "what_happened": "Normal trading conditions",
        "key_feature": "Low stress across all metrics"
    }
]

# Weights for distance calculation
FINGERPRINT_WEIGHTS = {
    "amri": 1.0,
    "bubble": 1.0,
    "contagion": 1.2,  # Higher weight - key indicator
    "correlations": 0.8,
    "clusters": 0.6
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Fingerprint:
    """Market fingerprint."""
    amri: float
    bubble: float
    contagion: float
    correlations: float
    clusters: int


@dataclass
class EpisodeMatch:
    """Historical episode match."""
    name: str
    date: str
    distance: float
    pattern: str
    what_happened: str
    key_feature: str
    fingerprint: Fingerprint


@dataclass
class FingerprintData:
    """Complete fingerprint analysis."""
    date: str
    current: Fingerprint
    nearest_match: EpisodeMatch
    match_quality: str
    key_differences: List[str]
    top_matches: List[EpisodeMatch]
    interpretation: str


# =============================================================================
# FINGERPRINT CALCULATOR
# =============================================================================

class FingerprintLibraryCalculator:
    """
    Compare current market state to historical episodes.
    
    Usage:
        calc = FingerprintLibraryCalculator()
        data = calc.calculate()
        calc.save_to_supabase(data)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize calculator."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        self.episodes = HISTORICAL_EPISODES
        self.weights = FINGERPRINT_WEIGHTS
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_current_fingerprint(self) -> Fingerprint:
        """Fetch current market fingerprint from various dials."""
        amri = 38.0
        bubble = 57.0
        contagion = 50.0
        correlations = 0.50
        clusters = 7
        
        if not self.supabase:
            return Fingerprint(amri, bubble, contagion, correlations, clusters)
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # AMRI
            response = self.supabase.table("amri_daily") \
                .select("amri") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                amri = float(response.data[0].get("amri", 38))
            
            # Bubble Index
            response = self.supabase.table("bubble_index_daily") \
                .select("bubble_index") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                bubble = float(response.data[0].get("bubble_index", 57))
            
            # Hypergraph contagion
            response = self.supabase.table("hypergraph_signals") \
                .select("contagion_score") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                contagion = float(response.data[0].get("contagion_score", 50))
            
            # Correlations
            response = self.supabase.table("correlation_daily") \
                .select("corr_20d") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                correlations = float(response.data[0].get("corr_20d", 0.50))
            
            # Clusters
            response = self.supabase.table("cluster_dial_daily") \
                .select("cluster_count") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                clusters = int(response.data[0].get("cluster_count", 7))
                
        except Exception as e:
            logger.warning(f"Error fetching fingerprint data: {e}")
        
        return Fingerprint(
            amri=amri,
            bubble=bubble,
            contagion=contagion,
            correlations=correlations,
            clusters=clusters
        )
    
    # =========================================================================
    # DISTANCE CALCULATION
    # =========================================================================
    
    def calculate_distance(self, current: Fingerprint, historical: Dict) -> float:
        """
        Calculate normalized Euclidean distance between fingerprints.
        
        Returns distance on 0-100 scale (0 = perfect match).
        """
        fp = historical["fingerprint"]
        weights = self.weights
        
        sum_squares = 0.0
        total_weight = 0.0
        
        # AMRI (0-100)
        amri_diff = (current.amri - fp["amri"]) / 100
        sum_squares += weights["amri"] * amri_diff ** 2
        total_weight += weights["amri"]
        
        # Bubble (0-100)
        bubble_diff = (current.bubble - fp["bubble"]) / 100
        sum_squares += weights["bubble"] * bubble_diff ** 2
        total_weight += weights["bubble"]
        
        # Contagion (0-100)
        contagion_diff = (current.contagion - fp["contagion"]) / 100
        sum_squares += weights["contagion"] * contagion_diff ** 2
        total_weight += weights["contagion"]
        
        # Correlations (0-1)
        corr_diff = current.correlations - fp["correlations"]
        sum_squares += weights["correlations"] * corr_diff ** 2
        total_weight += weights["correlations"]
        
        # Clusters (invert: fewer = more stress)
        # 3 clusters = 100 stress, 15 clusters = 0 stress
        current_cluster_stress = max(0, (15 - current.clusters) / 12 * 100)
        historical_cluster_stress = max(0, (15 - fp["clusters"]) / 12 * 100)
        cluster_diff = (current_cluster_stress - historical_cluster_stress) / 100
        sum_squares += weights["clusters"] * cluster_diff ** 2
        total_weight += weights["clusters"]
        
        # Normalized distance (0-100)
        distance = math.sqrt(sum_squares / total_weight) * 100
        
        return round(distance, 1)
    
    # =========================================================================
    # EPISODE MATCHING
    # =========================================================================
    
    def find_nearest_episode(self, current: Fingerprint) -> Tuple[EpisodeMatch, List[EpisodeMatch], str, List[str]]:
        """
        Find nearest historical episode match.
        
        Returns: (nearest_match, top_matches, match_quality, key_differences)
        """
        all_matches = []
        
        for episode in self.episodes:
            distance = self.calculate_distance(current, episode)
            fp = episode["fingerprint"]
            
            match = EpisodeMatch(
                name=episode["name"],
                date=episode["date"],
                distance=distance,
                pattern=episode["pattern"],
                what_happened=episode["what_happened"],
                key_feature=episode["key_feature"],
                fingerprint=Fingerprint(
                    amri=fp["amri"],
                    bubble=fp["bubble"],
                    contagion=fp["contagion"],
                    correlations=fp["correlations"],
                    clusters=fp["clusters"]
                )
            )
            all_matches.append(match)
        
        # Sort by distance
        all_matches.sort(key=lambda x: x.distance)
        
        nearest = all_matches[0]
        
        # Match quality
        if nearest.distance < 15:
            match_quality = "STRONG"
        elif nearest.distance < 30:
            match_quality = "MODERATE"
        elif nearest.distance < 50:
            match_quality = "WEAK"
        else:
            match_quality = "UNIQUE"
        
        # Key differences
        key_differences = []
        fp = nearest.fingerprint
        
        if abs(current.amri - fp.amri) > 15:
            direction = "higher" if current.amri > fp.amri else "lower"
            key_differences.append(f"AMRI {direction} ({current.amri:.0f} vs {fp.amri})")
        
        if abs(current.bubble - fp.bubble) > 15:
            direction = "higher" if current.bubble > fp.bubble else "lower"
            key_differences.append(f"Bubble {direction} ({current.bubble:.0f} vs {fp.bubble})")
        
        if abs(current.contagion - fp.contagion) > 15:
            direction = "higher" if current.contagion > fp.contagion else "lower"
            key_differences.append(f"Contagion {direction} ({current.contagion:.0f} vs {fp.contagion})")
        
        if abs(current.correlations - fp.correlations) > 0.15:
            direction = "higher" if current.correlations > fp.correlations else "lower"
            key_differences.append(f"Correlations {direction} ({current.correlations*100:.0f}% vs {fp.correlations*100:.0f}%)")
        
        return nearest, all_matches[:3], match_quality, key_differences
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> FingerprintData:
        """
        Calculate fingerprint analysis.
        
        Returns:
            FingerprintData with episode matching
        """
        logger.info("Calculating fingerprint analysis...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Get current fingerprint
        current = self.fetch_current_fingerprint()
        
        # Find nearest match
        nearest, top_matches, match_quality, key_differences = self.find_nearest_episode(current)
        
        # Generate interpretation
        if match_quality == "STRONG":
            interpretation = f"Current conditions closely match {nearest.name}. Pattern: {nearest.pattern}. Historical outcome: {nearest.what_happened}"
        elif match_quality == "MODERATE":
            interpretation = f"Some similarity to {nearest.name}, but notable differences. Monitor {nearest.key_feature}."
        elif match_quality == "WEAK":
            interpretation = f"Weak match to {nearest.name}. Current conditions may be transitional."
        else:
            interpretation = "Current conditions don't match known patterns. Unique market environment."
        
        logger.info(f"Nearest match: {nearest.name} ({match_quality})")
        logger.info(f"Distance: {nearest.distance}")
        
        return FingerprintData(
            date=date_str,
            current=current,
            nearest_match=nearest,
            match_quality=match_quality,
            key_differences=key_differences,
            top_matches=top_matches,
            interpretation=interpretation
        )
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_to_supabase(self, data: FingerprintData) -> Optional[Dict]:
        """Save fingerprint data to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return None
        
        record = {
            "date": data.date,
            # Current fingerprint
            "current_amri": data.current.amri,
            "current_bubble": data.current.bubble,
            "current_contagion": data.current.contagion,
            "current_correlations": data.current.correlations,
            "current_clusters": data.current.clusters,
            # Match
            "nearest_match": data.nearest_match.name,
            "match_date": data.nearest_match.date,
            "match_distance": data.nearest_match.distance,
            "match_quality": data.match_quality,
            "match_pattern": data.nearest_match.pattern,
            # Analysis
            "key_differences": data.key_differences,
            "interpretation": data.interpretation,
        }
        
        try:
            response = self.supabase.table("fingerprint_daily") \
                .upsert(record, on_conflict="date") \
                .execute()
            
            if response.data:
                logger.info(f"Saved fingerprint for {data.date}")
                return response.data[0]
        except Exception as e:
            logger.error(f"Failed to save fingerprint: {e}")
        
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run fingerprint analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Fingerprint Library")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    calc = FingerprintLibraryCalculator()
    data = calc.calculate()
    
    print(f"\n{'='*60}")
    print(f"FINGERPRINT ANALYSIS")
    print(f"{'='*60}")
    print(f"\nCurrent Fingerprint:")
    print(f"  AMRI: {data.current.amri:.1f}")
    print(f"  Bubble: {data.current.bubble:.1f}")
    print(f"  Contagion: {data.current.contagion:.1f}")
    print(f"  Correlations: {data.current.correlations*100:.0f}%")
    print(f"  Clusters: {data.current.clusters}")
    
    print(f"\nNearest Match: {data.nearest_match.name} ({data.nearest_match.date})")
    print(f"Distance: {data.nearest_match.distance:.1f} ({data.match_quality})")
    print(f"Pattern: {data.nearest_match.pattern}")
    print(f"Key Feature: {data.nearest_match.key_feature}")
    
    if data.key_differences:
        print(f"\nKey Differences:")
        for diff in data.key_differences:
            print(f"  - {diff}")
    
    print(f"\nTop 3 Matches:")
    for i, match in enumerate(data.top_matches, 1):
        print(f"  {i}. {match.name} (distance: {match.distance:.1f})")
    
    print(f"\nInterpretation: {data.interpretation}")
    print(f"{'='*60}\n")
    
    if args.save:
        result = calc.save_to_supabase(data)
        if result:
            print("Saved to Supabase")
    
    return data


if __name__ == "__main__":
    main()
