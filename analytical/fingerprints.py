"""
Historical Fingerprint Matching

Compares current market conditions to historical episodes
(COVID crash, 2022 tech, SVB crisis, etc.) to identify patterns.
"""

import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from .config import HISTORICAL_EPISODES, HistoricalEpisode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FingerprintMatch:
    """Result of fingerprint matching"""
    episode: str              # Name of matched episode
    match_score: float        # 0-100 match percentage
    pattern_type: str         # FAST_BREAK, SLOW_GRIND, V_RECOVERY
    quality: str              # HIGH, MEDIUM, LOW
    days_to_peak: int         # How many days the episode took
    characteristics: Dict[str, float]  # Matching characteristics
    display: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class FingerprintMatcher:
    """
    Match current conditions to historical episodes.
    """
    
    def __init__(self):
        self.episodes = HISTORICAL_EPISODES
    
    def calculate_match_score(
        self,
        current_amri: float,
        current_contagion: float,
        current_vix: float,
        cluster_speed: float = 0.5,
        episode: HistoricalEpisode = None
    ) -> float:
        """
        Calculate how closely current conditions match an episode.
        """
        if not episode:
            return 0.0
        
        # Normalize current values to episode peaks
        amri_ratio = min(current_amri / episode.peak_amri, 1.0)
        contagion_ratio = min(current_contagion / episode.peak_contagion, 1.0)
        vix_ratio = min(current_vix / episode.peak_vix, 1.0)
        
        # Compare to episode characteristics
        char = episode.characteristics
        speed_match = 1 - abs(cluster_speed - char.get("cluster_collapse_speed", 0.5))
        cross_match = 1 - abs(contagion_ratio - char.get("cross_pillar_surge", 0.5))
        vix_match = 1 - abs(vix_ratio - char.get("vix_acceleration", 0.5))
        
        # Weighted score
        score = (
            amri_ratio * 25 +
            contagion_ratio * 25 +
            speed_match * 20 +
            cross_match * 15 +
            vix_match * 15
        )
        
        return min(100, max(0, score))
    
    def find_best_match(
        self,
        current_amri: float,
        current_contagion: float,
        current_vix: float,
        cluster_speed: float = 0.5
    ) -> Tuple[HistoricalEpisode, float]:
        """Find the historical episode that best matches current conditions"""
        best_episode = None
        best_score = 0
        
        for episode in self.episodes:
            score = self.calculate_match_score(
                current_amri, current_contagion, current_vix,
                cluster_speed, episode
            )
            if score > best_score:
                best_score = score
                best_episode = episode
        
        return best_episode, best_score
    
    def determine_quality(self, match_score: float) -> str:
        """Classify match quality"""
        if match_score >= 70:
            return "HIGH"
        elif match_score >= 40:
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate(
        self,
        current_amri: float = 50.0,
        current_contagion: float = 50.0,
        current_vix: float = 18.0,
        cluster_speed: float = 0.5
    ) -> FingerprintMatch:
        """
        Find best matching historical episode.
        """
        episode, score = self.find_best_match(
            current_amri, current_contagion, current_vix, cluster_speed
        )
        
        if not episode or score < 20:
            return FingerprintMatch(
                episode="NONE",
                match_score=0,
                pattern_type="UNKNOWN",
                quality="LOW",
                days_to_peak=0,
                characteristics={},
                display="No clear pattern",
            )
        
        quality = self.determine_quality(score)
        
        display = f"{episode.name} ({score:.0f}%)"
        if quality == "HIGH":
            display = f"⚡ {display}"
        
        return FingerprintMatch(
            episode=episode.name,
            match_score=round(score, 1),
            pattern_type=episode.pattern_type,
            quality=quality,
            days_to_peak=episode.days_to_break,
            characteristics=episode.characteristics,
            display=display,
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    matcher = FingerprintMatcher()
    
    # Test with elevated conditions
    result = matcher.calculate(
        current_amri=72,
        current_contagion=65,
        current_vix=28,
        cluster_speed=0.7
    )
    
    print(f"\n{'='*60}")
    print("FINGERPRINT MATCHING")
    print(f"{'='*60}")
    print(f"Best Match: {result.episode}")
    print(f"Match Score: {result.match_score}%")
    print(f"Pattern: {result.pattern_type}")
    print(f"Quality: {result.quality}")
    print(f"Days to Peak: {result.days_to_peak}")
    print(f"Display: {result.display}")
