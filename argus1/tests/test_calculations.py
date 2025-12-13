"""
Unit Tests for Core Calculations

Tests cover:
- Bubble Index calculation
- Bifurcation Score formula
- Z-score calculations
- Regime determination
- Pillar averages
- Y2AI Index calculation
- Status determination (VALIDATING/NEUTRAL/CONTRADICTING)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# =============================================================================
# BUBBLE INDEX CALCULATION TESTS
# =============================================================================

class TestBubbleIndexCalculation:
    """Test Bubble Index formula: BI = (CAPE - 15) * 2.5 + 20"""
    
    def test_cheap_valuation(self):
        """CAPE 15 maps to Bubble Index 20 (cheap)"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        bi = calc.calculate_bubble_index(cape=15.0)
        
        assert bi == 20.0
    
    def test_fair_value(self):
        """CAPE 25 maps to Bubble Index 45 (fair value)"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        bi = calc.calculate_bubble_index(cape=25.0)
        
        assert bi == 45.0
    
    def test_expensive_valuation(self):
        """CAPE 35 maps to Bubble Index 70 (expensive)"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        bi = calc.calculate_bubble_index(cape=35.0)
        
        assert bi == 70.0
    
    def test_extreme_valuation(self):
        """CAPE 45 maps to Bubble Index 95 (extreme)"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        bi = calc.calculate_bubble_index(cape=45.0)
        
        assert bi == 95.0
    
    def test_clamped_at_zero(self):
        """Very low CAPE is clamped to Bubble Index 0"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        bi = calc.calculate_bubble_index(cape=5.0)  # Would be -5 unclamped
        
        assert bi == 0.0
    
    def test_clamped_at_100(self):
        """Very high CAPE is clamped to Bubble Index 100"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        bi = calc.calculate_bubble_index(cape=60.0)  # Would be 132.5 unclamped
        
        assert bi == 100.0


# =============================================================================
# BIFURCATION SCORE TESTS
# =============================================================================

class TestBifurcationScore:
    """Test Bifurcation Score formula: 0.6*BI_norm - 0.2*VIX_z - 0.2*Credit_z"""
    
    def test_neutral_inputs(self):
        """Neutral inputs (BI=50, z-scores=0) gives score near 0"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        score = calc.calculate_bifurcation_score(
            bubble_index=50.0,  # Normalized to 0
            vix_zscore=0.0,
            credit_zscore=0.0
        )
        
        assert score == 0.0
    
    def test_high_bubble_index_positive_score(self):
        """High Bubble Index (BI=100) gives positive contribution"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        score = calc.calculate_bifurcation_score(
            bubble_index=100.0,  # Normalized to +1
            vix_zscore=0.0,
            credit_zscore=0.0
        )
        
        # 0.6 * 1.0 = 0.6
        assert score == 0.6
    
    def test_low_bubble_index_negative_score(self):
        """Low Bubble Index (BI=0) gives negative contribution"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        score = calc.calculate_bifurcation_score(
            bubble_index=0.0,  # Normalized to -1
            vix_zscore=0.0,
            credit_zscore=0.0
        )
        
        # 0.6 * (-1.0) = -0.6
        assert score == -0.6
    
    def test_high_vix_reduces_score(self):
        """High VIX z-score reduces bifurcation score"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        score = calc.calculate_bifurcation_score(
            bubble_index=50.0,
            vix_zscore=2.0,  # High volatility
            credit_zscore=0.0
        )
        
        # 0.6*0 - 0.2*2.0 - 0.2*0 = -0.4
        assert score == -0.4
    
    def test_high_credit_spread_reduces_score(self):
        """High credit spread z-score reduces bifurcation score"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        score = calc.calculate_bifurcation_score(
            bubble_index=50.0,
            vix_zscore=0.0,
            credit_zscore=2.0  # Wide spreads
        )
        
        # 0.6*0 - 0.2*0 - 0.2*2.0 = -0.4
        assert score == -0.4
    
    def test_combined_stress_signals(self):
        """Combined stress signals give more negative score"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        score = calc.calculate_bifurcation_score(
            bubble_index=50.0,
            vix_zscore=2.0,
            credit_zscore=2.0
        )
        
        # 0.6*0 - 0.2*2.0 - 0.2*2.0 = -0.8
        assert score == -0.8
    
    def test_infrastructure_regime_score(self):
        """Typical infrastructure regime values"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        # High valuations, low volatility, tight spreads
        score = calc.calculate_bifurcation_score(
            bubble_index=75.0,  # Normalized to +0.5
            vix_zscore=-0.5,   # Below average VIX
            credit_zscore=-0.5  # Tight spreads
        )
        
        # 0.6*0.5 - 0.2*(-0.5) - 0.2*(-0.5) = 0.3 + 0.1 + 0.1 = 0.5
        assert abs(score - 0.5) < 0.01


# =============================================================================
# Z-SCORE CALCULATION TESTS
# =============================================================================

class TestZScoreCalculation:
    """Test z-score calculation from historical data"""
    
    def test_zscore_at_mean(self):
        """Value at mean has z-score of 0"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        history = pd.Series([10, 20, 30, 40, 50])  # Mean = 30
        
        zscore = calc._calculate_zscore(30.0, history)
        
        assert abs(zscore) < 0.01
    
    def test_zscore_one_std_above(self):
        """Value one std above mean has z-score of ~1"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        # Create series with known mean and std
        history = pd.Series([0, 0, 0, 0, 10])  # Mean=2, std≈4.47
        mean = history.mean()
        std = history.std()
        
        zscore = calc._calculate_zscore(mean + std, history)
        
        assert abs(zscore - 1.0) < 0.01
    
    def test_zscore_one_std_below(self):
        """Value one std below mean has z-score of ~-1"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        history = pd.Series([0, 0, 0, 0, 10])
        mean = history.mean()
        std = history.std()
        
        zscore = calc._calculate_zscore(mean - std, history)
        
        assert abs(zscore - (-1.0)) < 0.01
    
    def test_zscore_empty_history(self):
        """Empty history returns z-score of 0"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        history = pd.Series([])
        
        zscore = calc._calculate_zscore(100.0, history)
        
        assert zscore == 0.0
    
    def test_zscore_zero_std(self):
        """Zero standard deviation returns z-score of 0"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        history = pd.Series([50, 50, 50, 50])  # No variance
        
        zscore = calc._calculate_zscore(60.0, history)
        
        assert zscore == 0.0


# =============================================================================
# REGIME DETERMINATION TESTS
# =============================================================================

class TestRegimeDetermination:
    """Test regime classification based on bifurcation score"""
    
    def test_infrastructure_regime(self):
        """Score > 0.5 is INFRASTRUCTURE"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        regime = calc.determine_regime(bifurcation_score=0.6, vix=15.0)
        
        assert regime == "INFRASTRUCTURE"
    
    def test_adoption_regime(self):
        """Score 0.2 to 0.5 is ADOPTION"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        
        assert calc.determine_regime(0.3, vix=15.0) == "ADOPTION"
        assert calc.determine_regime(0.5, vix=15.0) == "ADOPTION"
        assert calc.determine_regime(0.2, vix=15.0) == "TRANSITION"  # Boundary
    
    def test_transition_regime(self):
        """Score -0.2 to 0.2 is TRANSITION"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        
        assert calc.determine_regime(0.0, vix=15.0) == "TRANSITION"
        assert calc.determine_regime(0.1, vix=15.0) == "TRANSITION"
        assert calc.determine_regime(-0.1, vix=15.0) == "TRANSITION"
    
    def test_bubble_warning_regime(self):
        """Score < -0.2 is BUBBLE_WARNING"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        
        assert calc.determine_regime(-0.3, vix=15.0) == "BUBBLE_WARNING"
        assert calc.determine_regime(-0.5, vix=15.0) == "BUBBLE_WARNING"
    
    def test_high_vix_forces_transition(self):
        """VIX > 30 forces TRANSITION regardless of score"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        
        # Even with high score, VIX override applies
        assert calc.determine_regime(0.8, vix=35.0) == "TRANSITION"
        assert calc.determine_regime(-0.5, vix=35.0) == "TRANSITION"


# =============================================================================
# STOCK TRACKER PILLAR TESTS
# =============================================================================

class TestPillarCalculations:
    """Test pillar average calculations"""
    
    def test_pillar_average_calculation(self):
        """Pillar averages are calculated correctly"""
        from y2ai.stock_tracker_enhanced import StockTracker, PILLARS
        
        tracker = StockTracker()
        
        # Mock stock data
        stock_data = {
            "TSM": {"change_today": 1.0, "change_5day": 2.0, "change_ytd": 10.0},
            "ASML": {"change_today": 2.0, "change_5day": 4.0, "change_ytd": 20.0},
            "VRT": {"change_today": 3.0, "change_5day": 6.0, "change_ytd": 30.0},
        }
        
        pillar = tracker._calculate_pillar_performance(stock_data, "supply_constraint")
        
        assert pillar.avg_today == 2.0  # (1+2+3)/3
        assert pillar.avg_5day == 4.0   # (2+4+6)/3
        assert pillar.avg_ytd == 20.0   # (10+20+30)/3
    
    def test_pillar_with_missing_stock(self):
        """Pillar calculates with available stocks only"""
        from y2ai.stock_tracker_enhanced import StockTracker
        
        tracker = StockTracker()
        
        # Only 2 of 3 stocks available
        stock_data = {
            "TSM": {"change_today": 1.0, "change_5day": 2.0, "change_ytd": 10.0},
            "ASML": {"change_today": 3.0, "change_5day": 4.0, "change_ytd": 20.0},
            # VRT missing
        }
        
        pillar = tracker._calculate_pillar_performance(stock_data, "supply_constraint")
        
        assert pillar.avg_today == 2.0  # (1+3)/2
    
    def test_pillar_all_missing(self):
        """Pillar returns None when all stocks missing"""
        from y2ai.stock_tracker_enhanced import StockTracker
        
        tracker = StockTracker()
        
        stock_data = {}  # No stocks
        
        pillar = tracker._calculate_pillar_performance(stock_data, "supply_constraint")
        
        assert pillar is None


# =============================================================================
# Y2AI INDEX TESTS
# =============================================================================

class TestY2AIIndex:
    """Test Y2AI Index (equal-weight of 9 pillar stocks)"""
    
    def test_y2ai_index_calculation(self):
        """Y2AI Index is equal-weight average of 9 pillar stocks"""
        from y2ai.stock_tracker_enhanced import StockTracker
        
        tracker = StockTracker()
        
        # All 9 pillar stocks
        stock_data = {
            # Supply Constraint
            "TSM": {"change_today": 1.0, "change_5day": 1.0, "change_ytd": 10.0},
            "ASML": {"change_today": 2.0, "change_5day": 2.0, "change_ytd": 20.0},
            "VRT": {"change_today": 3.0, "change_5day": 3.0, "change_ytd": 30.0},
            # Capital Efficiency
            "GOOGL": {"change_today": 4.0, "change_5day": 4.0, "change_ytd": 40.0},
            "MSFT": {"change_today": 5.0, "change_5day": 5.0, "change_ytd": 50.0},
            "AMZN": {"change_today": 6.0, "change_5day": 6.0, "change_ytd": 60.0},
            # Demand Depth
            "NVDA": {"change_today": 7.0, "change_5day": 7.0, "change_ytd": 70.0},
            "SNOW": {"change_today": 8.0, "change_5day": 8.0, "change_ytd": 80.0},
            "NOW": {"change_today": 9.0, "change_5day": 9.0, "change_ytd": 90.0},
        }
        
        y2ai = tracker._calculate_y2ai_index(stock_data)
        
        assert y2ai["today"] == 5.0   # (1+2+3+4+5+6+7+8+9)/9
        assert y2ai["5day"] == 5.0
        assert y2ai["ytd"] == 50.0
    
    def test_y2ai_index_partial_data(self):
        """Y2AI Index handles missing stocks"""
        from y2ai.stock_tracker_enhanced import StockTracker
        
        tracker = StockTracker()
        
        # Only 3 stocks available
        stock_data = {
            "TSM": {"change_today": 3.0, "change_5day": 3.0, "change_ytd": 30.0},
            "ASML": {"change_today": 6.0, "change_5day": 6.0, "change_ytd": 60.0},
            "VRT": {"change_today": 9.0, "change_5day": 9.0, "change_ytd": 90.0},
        }
        
        y2ai = tracker._calculate_y2ai_index(stock_data)
        
        assert y2ai["today"] == 6.0  # (3+6+9)/3


# =============================================================================
# STATUS DETERMINATION TESTS
# =============================================================================

class TestStatusDetermination:
    """Test VALIDATING/NEUTRAL/CONTRADICTING status"""
    
    def test_validating_status(self):
        """Y2AI outperforming SPY by >0.25% is VALIDATING"""
        from y2ai.stock_tracker_enhanced import StockTracker
        
        tracker = StockTracker()
        
        status = tracker._determine_status(y2ai_today=1.0, spy_today=0.5)
        
        assert status == "VALIDATING"
    
    def test_contradicting_status(self):
        """Y2AI underperforming SPY by >0.25% is CONTRADICTING"""
        from y2ai.stock_tracker_enhanced import StockTracker
        
        tracker = StockTracker()
        
        status = tracker._determine_status(y2ai_today=0.5, spy_today=1.0)
        
        assert status == "CONTRADICTING"
    
    def test_neutral_status(self):
        """Y2AI within ±0.25% of SPY is NEUTRAL"""
        from y2ai.stock_tracker_enhanced import StockTracker
        
        tracker = StockTracker()
        
        assert tracker._determine_status(1.0, 1.0) == "NEUTRAL"
        assert tracker._determine_status(1.0, 1.2) == "NEUTRAL"
        assert tracker._determine_status(1.0, 0.8) == "NEUTRAL"
    
    def test_boundary_cases(self):
        """Boundary cases at ±0.25%"""
        from y2ai.stock_tracker_enhanced import StockTracker
        
        tracker = StockTracker()
        
        # Exactly at boundary
        assert tracker._determine_status(1.25, 1.0) == "NEUTRAL"  # Exactly +0.25
        assert tracker._determine_status(0.75, 1.0) == "NEUTRAL"  # Exactly -0.25
        
        # Just over boundary
        assert tracker._determine_status(1.26, 1.0) == "VALIDATING"
        assert tracker._determine_status(0.74, 1.0) == "CONTRADICTING"


# =============================================================================
# DATA QUALITY TESTS
# =============================================================================

class TestDataQuality:
    """Test data quality scoring"""
    
    def test_bubble_index_data_quality(self):
        """Bubble index tracks data source quality"""
        from y2ai.bubble_index_enhanced import BubbleIndexReading
        
        # All live data
        reading_live = BubbleIndexReading(
            date="2025-01-01",
            vix=15.0, cape=30.0,
            credit_spread_ig=100.0, credit_spread_hy=350.0,
            vix_zscore=0.0, cape_zscore=0.0, credit_zscore=0.0,
            bubble_index=50.0, bifurcation_score=0.0,
            regime="TRANSITION",
            data_sources={"vix": "live", "cape": "live", "credit": "live"}
        )
        
        assert reading_live.data_quality_score == 1.0
        
        # Mixed live/fallback
        reading_mixed = BubbleIndexReading(
            date="2025-01-01",
            vix=15.0, cape=30.0,
            credit_spread_ig=100.0, credit_spread_hy=350.0,
            vix_zscore=0.0, cape_zscore=0.0, credit_zscore=0.0,
            bubble_index=50.0, bifurcation_score=0.0,
            regime="TRANSITION",
            data_sources={"vix": "live", "cape": "fallback", "credit": "live"}
        )
        
        assert abs(reading_mixed.data_quality_score - 0.667) < 0.01
    
    def test_stock_report_data_quality(self):
        """Stock report tracks fetch success rate"""
        from y2ai.stock_tracker_enhanced import DailyReport
        
        report = DailyReport(
            date="2025-01-01",
            stocks=[], pillars=[],
            y2ai_index_today=0, y2ai_index_5day=0, y2ai_index_ytd=0,
            spy_today=0, spy_5day=0, spy_ytd=0,
            qqq_today=0, qqq_5day=0, qqq_ytd=0,
            status="NEUTRAL",
            best_stock="NVDA", worst_stock="INTC",
            best_pillar="Supply", worst_pillar="Demand",
            stocks_fetched=15,
            stocks_failed=5,
        )
        
        assert report.data_quality_score == 0.75  # 15/20


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
