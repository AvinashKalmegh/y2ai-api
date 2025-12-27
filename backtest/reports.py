"""
Report Generator for ARGUS-1 Backtesting

Generates HTML and text reports from backtest results.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

from .performance import BacktestMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate backtest reports in various formats"""
    
    @staticmethod
    def generate_html(metrics: BacktestMetrics, signals_df=None) -> str:
        """
        Generate HTML report from backtest metrics.
        
        Args:
            metrics: BacktestMetrics from backtest run
            signals_df: Optional DataFrame of signals
        
        Returns:
            HTML string
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ARGUS-1 Backtest Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4361ee; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #4361ee; }}
        .metric-card.good {{ border-left-color: #4caf50; }}
        .metric-card.bad {{ border-left-color: #f44336; }}
        .metric-card.warn {{ border-left-color: #ff9800; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #1a1a2e; }}
        .metric-sub {{ font-size: 12px; color: #888; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .regime-normal {{ color: #4caf50; }}
        .regime-elevated {{ color: #2196f3; }}
        .regime-tension {{ color: #ff9800; }}
        .regime-fragile {{ color: #f44336; }}
        .regime-break {{ color: #9c27b0; }}
        .confusion-matrix {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; max-width: 400px; margin: 20px 0; }}
        .cm-cell {{ padding: 15px; text-align: center; border-radius: 4px; }}
        .cm-tp {{ background: #c8e6c9; }}
        .cm-tn {{ background: #c8e6c9; }}
        .cm-fp {{ background: #ffcdd2; }}
        .cm-fn {{ background: #ffcdd2; }}
        .summary {{ background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #888; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ARGUS-1 Backtest Report</h1>
        <p><strong>Period:</strong> {metrics.date_range}</p>
        <p><strong>Total Signals:</strong> {metrics.total_signals}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Key Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card {'good' if metrics.confusion.accuracy > 0.7 else 'warn' if metrics.confusion.accuracy > 0.5 else 'bad'}">
                <div class="metric-label">Overall Accuracy</div>
                <div class="metric-value">{metrics.confusion.accuracy:.1%}</div>
                <div class="metric-sub">Risk prediction accuracy</div>
            </div>
            <div class="metric-card {'good' if metrics.confusion.precision > 0.6 else 'warn' if metrics.confusion.precision > 0.4 else 'bad'}">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{metrics.confusion.precision:.1%}</div>
                <div class="metric-sub">When we call risk, it happens</div>
            </div>
            <div class="metric-card {'good' if metrics.confusion.recall > 0.7 else 'warn' if metrics.confusion.recall > 0.5 else 'bad'}">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{metrics.confusion.recall:.1%}</div>
                <div class="metric-sub">Drawdowns we caught</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{metrics.confusion.f1_score:.3f}</div>
                <div class="metric-sub">Harmonic mean of P & R</div>
            </div>
        </div>
        
        <h2>Correlations</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">AMRI vs Max Drawdown</div>
                <div class="metric-value">{metrics.amri_dd_correlation:.3f}</div>
                <div class="metric-sub">Higher AMRI → larger drawdown?</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">AMRI vs Forward Return</div>
                <div class="metric-value">{metrics.amri_return_correlation:.3f}</div>
                <div class="metric-sub">Higher AMRI → lower return?</div>
            </div>
        </div>
        
        <h2>Regime Distribution</h2>
        <table>
            <tr>
                <th>Regime</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            {''.join(f'''
            <tr>
                <td class="regime-{regime.lower()}">{regime}</td>
                <td>{count}</td>
                <td>{metrics.regime_pct.get(regime, 0)*100:.1f}%</td>
            </tr>
            ''' for regime, count in sorted(metrics.regime_counts.items()))}
        </table>
        
        <h2>Regime Performance (20D Outcomes)</h2>
        <table>
            <tr>
                <th>Regime</th>
                <th>Count</th>
                <th>Avg Return</th>
                <th>Avg Max DD</th>
                <th>>5% DD Rate</th>
                <th>>10% DD Rate</th>
            </tr>
            {''.join(f'''
            <tr>
                <td class="regime-{regime.lower()}">{regime}</td>
                <td>{perf.count}</td>
                <td style="color: {'#4caf50' if perf.avg_return_20d > 0 else '#f44336'}">{perf.avg_return_20d:+.2f}%</td>
                <td style="color: {'#4caf50' if perf.avg_max_dd_20d > -5 else '#f44336'}">{perf.avg_max_dd_20d:.2f}%</td>
                <td>{perf.pct_with_5pct_dd:.1f}%</td>
                <td>{perf.pct_with_10pct_dd:.1f}%</td>
            </tr>
            ''' for regime, perf in sorted(metrics.regime_performance.items()))}
        </table>
        
        <h2>Confusion Matrix</h2>
        <p>Predicted Risk = TENSION, FRAGILE, BREAK, or VETO active<br>
        Actual Bad = >5% drawdown in 20 days</p>
        <div class="confusion-matrix">
            <div class="cm-cell cm-tp">
                <strong>True Positive</strong><br>
                {metrics.confusion.true_positives}<br>
                <small>Predicted risk, got drawdown</small>
            </div>
            <div class="cm-cell cm-fp">
                <strong>False Positive</strong><br>
                {metrics.confusion.false_positives}<br>
                <small>Predicted risk, market OK</small>
            </div>
            <div class="cm-cell cm-fn">
                <strong>False Negative</strong><br>
                {metrics.confusion.false_negatives}<br>
                <small>Predicted safe, got drawdown</small>
            </div>
            <div class="cm-cell cm-tn">
                <strong>True Negative</strong><br>
                {metrics.confusion.true_negatives}<br>
                <small>Predicted safe, market OK</small>
            </div>
        </div>
        
        <h2>VETO Signal Analysis</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">VETO Accuracy</div>
                <div class="metric-value">{metrics.veto_accuracy:.1%}</div>
                <div class="metric-sub">When VETO fires, drawdown follows</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg DD with VETO</div>
                <div class="metric-value">{metrics.veto_avg_dd:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg DD without VETO</div>
                <div class="metric-value">{metrics.non_veto_avg_dd:.2f}%</div>
            </div>
        </div>
        
        <h2>Signal Quality</h2>
        <div class="metric-grid">
            <div class="metric-card bad">
                <div class="metric-label">Missed Drawdowns</div>
                <div class="metric-value">{metrics.missed_drawdowns}</div>
                <div class="metric-sub">>10% DD not flagged</div>
            </div>
            <div class="metric-card warn">
                <div class="metric-label">False Alarms</div>
                <div class="metric-value">{metrics.false_alarms}</div>
                <div class="metric-sub">Risk signal, no drawdown</div>
            </div>
        </div>
        
        <div class="summary">
            <h3>Summary</h3>
            <p>
                Over the test period, ARGUS-1 achieved <strong>{metrics.confusion.accuracy:.1%}</strong> accuracy
                in predicting market risk. High-risk regime signals (TENSION/FRAGILE/BREAK) were followed by
                significant drawdowns <strong>{metrics.confusion.precision:.1%}</strong> of the time.
                The system successfully caught <strong>{metrics.confusion.recall:.1%}</strong> of all major drawdown events.
            </p>
            <p>
                AMRI showed a <strong>{metrics.amri_dd_correlation:.3f}</strong> correlation with subsequent drawdowns,
                {'indicating the metric has predictive value.' if abs(metrics.amri_dd_correlation) > 0.3 else 'suggesting room for improvement.'}
            </p>
        </div>
        
        <div class="footer">
            <p>Generated by ARGUS-1 Backtesting Module | Y2AI Research</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    @staticmethod
    def save_html(metrics: BacktestMetrics, filepath: str, signals_df=None):
        """Save HTML report to file"""
        html = ReportGenerator.generate_html(metrics, signals_df)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Saved HTML report to {filepath}")
    
    @staticmethod
    def generate_json(metrics: BacktestMetrics) -> str:
        """Generate JSON report"""
        data = {
            "generated_at": datetime.now().isoformat(),
            "period": metrics.date_range,
            "total_signals": metrics.total_signals,
            "accuracy": {
                "overall": metrics.confusion.accuracy,
                "precision": metrics.confusion.precision,
                "recall": metrics.confusion.recall,
                "f1_score": metrics.confusion.f1_score,
            },
            "confusion_matrix": {
                "true_positives": metrics.confusion.true_positives,
                "false_positives": metrics.confusion.false_positives,
                "true_negatives": metrics.confusion.true_negatives,
                "false_negatives": metrics.confusion.false_negatives,
            },
            "correlations": {
                "amri_drawdown": metrics.amri_dd_correlation,
                "amri_return": metrics.amri_return_correlation,
            },
            "regime_distribution": metrics.regime_counts,
            "regime_performance": {
                regime: {
                    "count": perf.count,
                    "avg_return_20d": perf.avg_return_20d,
                    "avg_max_dd_20d": perf.avg_max_dd_20d,
                    "pct_with_5pct_dd": perf.pct_with_5pct_dd,
                    "pct_with_10pct_dd": perf.pct_with_10pct_dd,
                }
                for regime, perf in metrics.regime_performance.items()
            },
            "veto_analysis": {
                "accuracy": metrics.veto_accuracy,
                "avg_dd_with_veto": metrics.veto_avg_dd,
                "avg_dd_without_veto": metrics.non_veto_avg_dd,
            },
            "signal_quality": {
                "missed_drawdowns": metrics.missed_drawdowns,
                "false_alarms": metrics.false_alarms,
            },
        }
        return json.dumps(data, indent=2)
    
    @staticmethod
    def save_json(metrics: BacktestMetrics, filepath: str):
        """Save JSON report to file"""
        json_str = ReportGenerator.generate_json(metrics)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)
        logger.info(f"Saved JSON report to {filepath}")