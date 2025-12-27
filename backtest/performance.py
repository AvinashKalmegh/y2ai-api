"""
Performance Analyzer for ARGUS-1 Backtesting

Compares ARGUS-1 regime signals against actual market outcomes
to evaluate prediction accuracy and signal quality.

Metrics:
- Regime-to-outcome accuracy
- AMRI-to-drawdown correlation
- Signal timing analysis
- False positive/negative rates
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .outcomes import MarketOutcome

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """Single signal record for analysis"""
    date: str
    regime: str
    amri: float
    veto_active: bool
    contagion: float
    sac: float
    outcome: Optional[MarketOutcome] = None


@dataclass 
class ConfusionMatrix:
    """Confusion matrix for regime predictions"""
    true_positives: int = 0   # Predicted risk, got drawdown
    false_positives: int = 0  # Predicted risk, market was fine
    true_negatives: int = 0   # Predicted safe, market was fine
    false_negatives: int = 0  # Predicted safe, got drawdown
    
    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class RegimePerformance:
    """Performance stats for a specific regime"""
    regime: str
    count: int = 0
    avg_return_20d: float = 0.0
    avg_max_dd_20d: float = 0.0
    pct_with_5pct_dd: float = 0.0
    pct_with_10pct_dd: float = 0.0
    avg_vol: float = 0.0


@dataclass
class BacktestMetrics:
    """Complete backtest performance metrics"""
    # Overall stats
    total_signals: int = 0
    date_range: str = ""
    
    # Regime distribution
    regime_counts: Dict[str, int] = field(default_factory=dict)
    regime_pct: Dict[str, float] = field(default_factory=dict)
    
    # Per-regime performance
    regime_performance: Dict[str, RegimePerformance] = field(default_factory=dict)
    
    # Confusion matrix (risk vs safe)
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    
    # AMRI correlation with outcomes
    amri_dd_correlation: float = 0.0
    amri_return_correlation: float = 0.0
    
    # VETO signal performance
    veto_accuracy: float = 0.0
    veto_avg_dd: float = 0.0
    non_veto_avg_dd: float = 0.0
    
    # Timing metrics
    avg_lead_time: float = 0.0  # Days before major drawdown
    missed_drawdowns: int = 0   # Major drawdowns not flagged
    false_alarms: int = 0       # Fragile/Break signals with no drawdown
    
    def summary(self) -> str:
        """Generate text summary of metrics"""
        lines = [
            "=" * 60,
            "ARGUS-1 BACKTEST RESULTS",
            "=" * 60,
            f"Period: {self.date_range}",
            f"Total Signals: {self.total_signals}",
            "",
            "REGIME DISTRIBUTION:",
        ]
        
        for regime, count in sorted(self.regime_counts.items()):
            pct = self.regime_pct.get(regime, 0) * 100
            lines.append(f"  {regime}: {count} ({pct:.1f}%)")
        
        lines.extend([
            "",
            "REGIME PERFORMANCE (20D outcomes):",
        ])
        
        for regime, perf in sorted(self.regime_performance.items()):
            lines.append(f"  {regime}:")
            lines.append(f"    Avg Return: {perf.avg_return_20d:.2f}%")
            lines.append(f"    Avg Max DD: {perf.avg_max_dd_20d:.2f}%")
            lines.append(f"    >5% DD rate: {perf.pct_with_5pct_dd:.1f}%")
            lines.append(f"    >10% DD rate: {perf.pct_with_10pct_dd:.1f}%")
        
        lines.extend([
            "",
            "PREDICTION ACCURACY:",
            f"  Overall Accuracy: {self.confusion.accuracy:.1%}",
            f"  Precision (risk calls): {self.confusion.precision:.1%}",
            f"  Recall (caught drawdowns): {self.confusion.recall:.1%}",
            f"  F1 Score: {self.confusion.f1_score:.3f}",
            "",
            "CORRELATIONS:",
            f"  AMRI vs Max Drawdown: {self.amri_dd_correlation:.3f}",
            f"  AMRI vs Forward Return: {self.amri_return_correlation:.3f}",
            "",
            "VETO SIGNAL:",
            f"  VETO Accuracy: {self.veto_accuracy:.1%}",
            f"  Avg DD when VETO active: {self.veto_avg_dd:.2f}%",
            f"  Avg DD when no VETO: {self.non_veto_avg_dd:.2f}%",
            "",
            "TIMING:",
            f"  Missed Major Drawdowns: {self.missed_drawdowns}",
            f"  False Alarms: {self.false_alarms}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class PerformanceAnalyzer:
    """
    Analyze ARGUS-1 backtest performance.
    
    Compares regime signals against actual market outcomes
    to measure prediction accuracy and signal quality.
    """
    
    # Regime risk classification
    HIGH_RISK_REGIMES = {"FRAGILE", "BREAK", "TENSION"}
    LOW_RISK_REGIMES = {"NORMAL", "ELEVATED"}
    
    # Outcome thresholds
    DRAWDOWN_THRESHOLD = -5.0  # % drawdown to classify as "bad"
    MAJOR_DRAWDOWN = -10.0     # % for major market event
    
    def __init__(self):
        self.signals: List[SignalRecord] = []
    
    def add_signal(self, date: str, regime: str, amri: float,
                   veto_active: bool, contagion: float, sac: float,
                   outcome: Optional[MarketOutcome] = None):
        """Add a signal record for analysis"""
        self.signals.append(SignalRecord(
            date=date,
            regime=regime,
            amri=amri,
            veto_active=veto_active,
            contagion=contagion,
            sac=sac,
            outcome=outcome
        ))
    
    def calculate_metrics(self) -> BacktestMetrics:
        """Calculate all performance metrics"""
        metrics = BacktestMetrics()
        
        if not self.signals:
            return metrics
        
        # Filter signals with outcomes
        valid_signals = [s for s in self.signals if s.outcome is not None]
        
        if not valid_signals:
            return metrics
        
        metrics.total_signals = len(valid_signals)
        metrics.date_range = f"{valid_signals[0].date} to {valid_signals[-1].date}"
        
        # Regime distribution
        for s in valid_signals:
            metrics.regime_counts[s.regime] = metrics.regime_counts.get(s.regime, 0) + 1
        
        for regime, count in metrics.regime_counts.items():
            metrics.regime_pct[regime] = count / len(valid_signals)
        
        # Per-regime performance
        metrics.regime_performance = self._calc_regime_performance(valid_signals)
        
        # Confusion matrix
        metrics.confusion = self._calc_confusion_matrix(valid_signals)
        
        # Correlations
        amris = [s.amri for s in valid_signals]
        returns_20d = [s.outcome.return_20d for s in valid_signals]
        max_dds = [s.outcome.max_dd_20d for s in valid_signals]
        
        metrics.amri_dd_correlation = self._correlation(amris, max_dds)
        metrics.amri_return_correlation = self._correlation(amris, returns_20d)
        
        # VETO analysis
        veto_signals = [s for s in valid_signals if s.veto_active]
        non_veto_signals = [s for s in valid_signals if not s.veto_active]
        
        if veto_signals:
            veto_bad = sum(1 for s in veto_signals if s.outcome.max_dd_20d < self.DRAWDOWN_THRESHOLD)
            metrics.veto_accuracy = veto_bad / len(veto_signals)
            metrics.veto_avg_dd = np.mean([s.outcome.max_dd_20d for s in veto_signals])
        
        if non_veto_signals:
            metrics.non_veto_avg_dd = np.mean([s.outcome.max_dd_20d for s in non_veto_signals])
        
        # Timing analysis
        metrics.missed_drawdowns = self._count_missed_drawdowns(valid_signals)
        metrics.false_alarms = self._count_false_alarms(valid_signals)
        
        return metrics
    
    def _calc_regime_performance(self, signals: List[SignalRecord]) -> Dict[str, RegimePerformance]:
        """Calculate performance stats per regime"""
        by_regime = defaultdict(list)
        for s in signals:
            by_regime[s.regime].append(s)
        
        performance = {}
        for regime, regime_signals in by_regime.items():
            perf = RegimePerformance(regime=regime)
            perf.count = len(regime_signals)
            
            returns = [s.outcome.return_20d for s in regime_signals]
            dds = [s.outcome.max_dd_20d for s in regime_signals]
            vols = [s.outcome.realized_vol_20d for s in regime_signals if s.outcome.realized_vol_20d > 0]
            
            perf.avg_return_20d = np.mean(returns) if returns else 0
            perf.avg_max_dd_20d = np.mean(dds) if dds else 0
            perf.pct_with_5pct_dd = 100 * sum(1 for d in dds if d < -5) / len(dds) if dds else 0
            perf.pct_with_10pct_dd = 100 * sum(1 for d in dds if d < -10) / len(dds) if dds else 0
            perf.avg_vol = np.mean(vols) if vols else 0
            
            performance[regime] = perf
        
        return performance
    
    def _calc_confusion_matrix(self, signals: List[SignalRecord]) -> ConfusionMatrix:
        """Calculate confusion matrix for risk prediction"""
        cm = ConfusionMatrix()
        
        for s in signals:
            predicted_risk = s.regime in self.HIGH_RISK_REGIMES or s.veto_active
            actual_bad = s.outcome.max_dd_20d < self.DRAWDOWN_THRESHOLD
            
            if predicted_risk and actual_bad:
                cm.true_positives += 1
            elif predicted_risk and not actual_bad:
                cm.false_positives += 1
            elif not predicted_risk and not actual_bad:
                cm.true_negatives += 1
            else:
                cm.false_negatives += 1
        
        return cm
    
    def _count_missed_drawdowns(self, signals: List[SignalRecord]) -> int:
        """Count major drawdowns not flagged by high-risk regime"""
        missed = 0
        for s in signals:
            if s.outcome.max_dd_20d < self.MAJOR_DRAWDOWN:
                # Major drawdown occurred
                if s.regime not in self.HIGH_RISK_REGIMES and not s.veto_active:
                    missed += 1
        return missed
    
    def _count_false_alarms(self, signals: List[SignalRecord]) -> int:
        """Count high-risk signals with no significant drawdown"""
        false_alarms = 0
        for s in signals:
            if s.regime in self.HIGH_RISK_REGIMES or s.veto_active:
                # We predicted risk
                if s.outcome.max_dd_20d > -3:  # Less than 3% drawdown
                    false_alarms += 1
        return false_alarms
    
    def _correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation"""
        if len(x) != len(y) or len(x) < 3:
            return 0.0
        
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        # Remove NaN
        mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
        x_arr = x_arr[mask]
        y_arr = y_arr[mask]
        
        if len(x_arr) < 3:
            return 0.0
        
        return np.corrcoef(x_arr, y_arr)[0, 1]
    
    def get_signals_df(self):
        """Export signals as DataFrame for further analysis"""
        import pandas as pd
        
        data = []
        for s in self.signals:
            row = {
                'date': s.date,
                'regime': s.regime,
                'amri': s.amri,
                'veto_active': s.veto_active,
                'contagion': s.contagion,
                'sac': s.sac,
            }
            if s.outcome:
                row.update({
                    'return_1d': s.outcome.return_1d,
                    'return_5d': s.outcome.return_5d,
                    'return_20d': s.outcome.return_20d,
                    'max_dd_20d': s.outcome.max_dd_20d,
                    'outcome_label': s.outcome.outcome_label,
                })
            data.append(row)
        
        return pd.DataFrame(data)
