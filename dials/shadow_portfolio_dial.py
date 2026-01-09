"""
SHADOW PORTFOLIO DIAL
=====================
Track portfolio positions and generate risk alerts.

Alert Sources:
1. Credit regime vs portfolio positioning
2. VIX regime (including Pre-Shock detection)
3. ETF vs FRED credit divergence
4. Breadth divergence
5. Correlation regime
6. Concentration risk

Risk Scoring:
- Credit regime: 0 (Healthy) to 4 (Crisis)
- VIX regime: 0 (Calm) to 3 (Crisis)
- Divergence alerts: +0.5 to +2.0
- Concentration: +0.5 to +1.0

Overall Risk:
- LOW: Score < 2
- ELEVATED: Score 2-4
- MODERATE: Score 4-6
- HIGH: Score > 6
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PORTFOLIO_CONFIG = {
    "credit_risk_scores": {
        "Healthy": 0,
        "Caution": 1,
        "Fragile": 2,
        "Stressed": 3,
        "Crisis": 4,
        "Unknown": 1
    },
    "vix_risk_scores": {
        "Calm": 0,
        "Normal": 0,
        "Elevated": 1,
        "Fragile": 1,
        "Pre-Shock": 2,
        "Crisis": 3,
    },
    "overall_risk_thresholds": {
        "low": 2,
        "elevated": 4,
        "moderate": 6,
        # > 6 = HIGH
    },
    "concentration_threshold": 0.40,  # Single pillar > 40%
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PortfolioPosition:
    """Single portfolio position."""
    ticker: str
    name: str
    pillar: str
    weight: float
    signal: str  # Overweight, Neutral, Underweight


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics."""
    total_positions: int
    total_weight: float
    overweight_exposure: float
    neutral_exposure: float
    underweight_exposure: float
    overweight_count: int
    neutral_count: int
    underweight_count: int
    max_pillar: str
    max_pillar_weight: float
    pillar_weights: Dict[str, float]


@dataclass
class Alert:
    """Risk alert."""
    level: str  # OK, CAUTION, WARNING, ALERT
    message: str


@dataclass
class ShadowPortfolioData:
    """Complete shadow portfolio data."""
    date: str
    # Regimes
    credit_regime: str
    vix_regime: str
    etf_credit_regime: str
    breadth_20d_regime: str
    breadth_50d_regime: str
    correlation_regime: str
    # Risk
    risk_score: float
    overall_risk: str
    # Metrics
    metrics: PortfolioMetrics
    # Alerts
    alerts: List[Alert]


# =============================================================================
# SHADOW PORTFOLIO CALCULATOR
# =============================================================================

class ShadowPortfolioCalculator:
    """
    Track portfolio and generate risk alerts.
    
    Usage:
        calc = ShadowPortfolioCalculator()
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
        
        self.config = PORTFOLIO_CONFIG
    
    # =========================================================================
    # REGIME FETCHING
    # =========================================================================
    
    def get_credit_regime(self) -> str:
        """Get credit regime from credit_spread_daily."""
        if not self.supabase:
            return "Unknown"
        
        try:
            response = self.supabase.table("credit_spread_daily") \
                .select("regime") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return response.data[0].get("regime", "Unknown")
        except:
            pass
        
        return "Unknown"
    
    def get_vix_regime(self) -> Dict:
        """Get VIX regime from vix_dial_daily."""
        result = {"regime": "Unknown", "vix": 0, "is_pre_shock": False}
        
        if not self.supabase:
            return result
        
        try:
            response = self.supabase.table("vix_dial_daily") \
                .select("regime, vix, bb_regime") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                d = response.data[0]
                result["regime"] = d.get("regime", "Unknown")
                result["vix"] = d.get("vix", 0)
                result["is_pre_shock"] = d.get("regime") == "Pre-Shock"
        except:
            pass
        
        return result
    
    def get_etf_credit_regime(self) -> str:
        """Get ETF credit regime (real-time spreads)."""
        if not self.supabase:
            return "Unknown"
        
        try:
            response = self.supabase.table("etf_dial_daily") \
                .select("credit_regime") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return response.data[0].get("credit_regime", "Unknown")
        except:
            pass
        
        return "Unknown"
    
    def get_breadth_regimes(self) -> Dict:
        """Get breadth regimes (20D and 50D)."""
        result = {"regime_20d": "Unknown", "regime_50d": "Unknown", "value_20d": 0, "value_50d": 0}
        
        if not self.supabase:
            return result
        
        try:
            response = self.supabase.table("breadth_daily") \
                .select("regime_20d, regime_50d, breadth_20d, breadth_50d") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                d = response.data[0]
                result["regime_20d"] = d.get("regime_20d", "Unknown")
                result["regime_50d"] = d.get("regime_50d", "Unknown")
                result["value_20d"] = d.get("breadth_20d", 0)
                result["value_50d"] = d.get("breadth_50d", 0)
        except:
            pass
        
        return result
    
    def get_correlation_regime(self) -> Dict:
        """Get correlation regime."""
        result = {"regime": "Unknown", "value": 0}
        
        if not self.supabase:
            return result
        
        try:
            response = self.supabase.table("correlation_daily") \
                .select("regime_20d, corr_20d") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                d = response.data[0]
                result["regime"] = d.get("regime_20d", "Unknown")
                result["value"] = d.get("corr_20d", 0)
        except:
            pass
        
        return result
    
    # =========================================================================
    # PORTFOLIO METRICS
    # =========================================================================
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get portfolio metrics from shadow_portfolio table."""
        default = PortfolioMetrics(
            total_positions=0, total_weight=0,
            overweight_exposure=0, neutral_exposure=0, underweight_exposure=0,
            overweight_count=0, neutral_count=0, underweight_count=0,
            max_pillar="", max_pillar_weight=0,
            pillar_weights={}
        )
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("shadow_portfolio") \
                .select("ticker, name, pillar, weight, signal") \
                .execute()
            
            if not response.data:
                return default
            
            positions = response.data
            
            total_weight = 0
            overweight_exposure = 0
            neutral_exposure = 0
            underweight_exposure = 0
            overweight_count = 0
            neutral_count = 0
            underweight_count = 0
            pillar_weights = {}
            
            for pos in positions:
                weight = float(pos.get("weight", 0))
                pillar = pos.get("pillar", "Unknown")
                signal = pos.get("signal", "Neutral")
                
                total_weight += weight
                
                if pillar not in pillar_weights:
                    pillar_weights[pillar] = 0
                pillar_weights[pillar] += weight
                
                if signal == "Overweight":
                    overweight_exposure += weight
                    overweight_count += 1
                elif signal == "Underweight":
                    underweight_exposure += weight
                    underweight_count += 1
                else:
                    neutral_exposure += weight
                    neutral_count += 1
            
            # Find max pillar
            max_pillar = ""
            max_weight = 0
            for pillar, weight in pillar_weights.items():
                if weight > max_weight:
                    max_weight = weight
                    max_pillar = pillar
            
            return PortfolioMetrics(
                total_positions=len(positions),
                total_weight=total_weight,
                overweight_exposure=overweight_exposure,
                neutral_exposure=neutral_exposure,
                underweight_exposure=underweight_exposure,
                overweight_count=overweight_count,
                neutral_count=neutral_count,
                underweight_count=underweight_count,
                max_pillar=max_pillar,
                max_pillar_weight=max_weight,
                pillar_weights=pillar_weights
            )
            
        except Exception as e:
            logger.error(f"Error getting portfolio metrics: {e}")
        
        return default
    
    # =========================================================================
    # ALERT GENERATION
    # =========================================================================
    
    def generate_alerts(
        self,
        credit_regime: str,
        vix_data: Dict,
        etf_credit_regime: str,
        breadth_data: Dict,
        correlation_data: Dict,
        metrics: PortfolioMetrics
    ) -> tuple:
        """
        Generate risk alerts and calculate risk score.
        
        Returns: (alerts, risk_score, overall_risk)
        """
        alerts = []
        risk_score = 0.0
        
        # 1. Credit regime risk
        risk_score += self.config["credit_risk_scores"].get(credit_regime, 1)
        
        # 2. VIX regime risk
        vix_regime = vix_data.get("regime", "Unknown")
        vix_score = self.config["vix_risk_scores"].get(vix_regime, 0)
        risk_score += vix_score
        
        if vix_regime == "Pre-Shock":
            alerts.append(Alert(
                level="WARNING",
                message="VIX Pre-Shock: Volatility compression detected - calm before potential storm"
            ))
        elif vix_regime == "Crisis":
            alerts.append(Alert(
                level="ALERT",
                message="VIX Crisis: Extreme volatility - maximum caution"
            ))
        elif vix_regime == "Fragile":
            alerts.append(Alert(
                level="CAUTION",
                message="VIX Fragile: Elevated volatility - reduce risk exposure"
            ))
        
        # 3. ETF vs FRED credit divergence
        if etf_credit_regime != "Unknown":
            fred_healthy = credit_regime in ["Healthy", "Caution"]
            etf_stressed = etf_credit_regime in ["Fragile", "Stressed"]
            
            if fred_healthy and etf_stressed:
                alerts.append(Alert(
                    level="WARNING",
                    message=f"Credit Divergence: ETF spreads ({etf_credit_regime}) leading FRED ({credit_regime}) - early warning"
                ))
                risk_score += 1.5
            
            if etf_credit_regime == "Stressed":
                alerts.append(Alert(
                    level="ALERT",
                    message="ETF Credit Stressed: Real-time spreads showing significant widening"
                ))
                risk_score += 2
        
        # 4. Breadth divergence (20D healthy but 50D fragile)
        regime_20d = breadth_data.get("regime_20d", "Unknown")
        regime_50d = breadth_data.get("regime_50d", "Unknown")
        
        if regime_20d == "Healthy" and regime_50d in ["Fragile", "Stressed"]:
            alerts.append(Alert(
                level="WARNING",
                message=f"Breadth Divergence: 20D healthy but 50D {regime_50d} - rally on thin ice"
            ))
            risk_score += 1.0
        
        # 5. Correlation risk
        corr_regime = correlation_data.get("regime", "Unknown")
        if corr_regime in ["Stressed", "Collapse"]:
            alerts.append(Alert(
                level="ALERT",
                message=f"Correlations {corr_regime}: Diversification failing"
            ))
            risk_score += 1.5
        elif corr_regime == "Fragile":
            alerts.append(Alert(
                level="CAUTION",
                message="Correlations elevated - watch for contagion"
            ))
            risk_score += 0.5
        
        # 6. Concentration risk
        if metrics.max_pillar_weight > self.config["concentration_threshold"]:
            alerts.append(Alert(
                level="CAUTION",
                message=f"Concentration: {metrics.max_pillar} at {metrics.max_pillar_weight*100:.1f}%"
            ))
            risk_score += 0.5
        
        # 7. Credit regime with high overweight exposure
        if credit_regime in ["Fragile", "Stressed", "Crisis"] and metrics.overweight_exposure > 0.5:
            alerts.append(Alert(
                level="WARNING",
                message=f"Risk mismatch: {metrics.overweight_exposure*100:.0f}% overweight during {credit_regime} credit"
            ))
            risk_score += 1.0
        
        # Overall risk level
        thresholds = self.config["overall_risk_thresholds"]
        if risk_score < thresholds["low"]:
            overall_risk = "LOW"
        elif risk_score < thresholds["elevated"]:
            overall_risk = "ELEVATED"
        elif risk_score < thresholds["moderate"]:
            overall_risk = "MODERATE"
        else:
            overall_risk = "HIGH"
        
        # Add OK alert if no issues
        if not alerts:
            alerts.append(Alert(level="OK", message="No significant risk alerts"))
        
        return alerts, risk_score, overall_risk
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> ShadowPortfolioData:
        """
        Calculate shadow portfolio data.
        
        Returns:
            ShadowPortfolioData with alerts and metrics
        """
        logger.info("Calculating shadow portfolio...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Get regimes
        credit_regime = self.get_credit_regime()
        vix_data = self.get_vix_regime()
        etf_credit_regime = self.get_etf_credit_regime()
        breadth_data = self.get_breadth_regimes()
        correlation_data = self.get_correlation_regime()
        
        # Get portfolio metrics
        metrics = self.get_portfolio_metrics()
        
        # Generate alerts
        alerts, risk_score, overall_risk = self.generate_alerts(
            credit_regime, vix_data, etf_credit_regime,
            breadth_data, correlation_data, metrics
        )
        
        logger.info(f"Portfolio Risk: {overall_risk} (score: {risk_score:.1f})")
        
        return ShadowPortfolioData(
            date=date_str,
            credit_regime=credit_regime,
            vix_regime=vix_data.get("regime", "Unknown"),
            etf_credit_regime=etf_credit_regime,
            breadth_20d_regime=breadth_data.get("regime_20d", "Unknown"),
            breadth_50d_regime=breadth_data.get("regime_50d", "Unknown"),
            correlation_regime=correlation_data.get("regime", "Unknown"),
            risk_score=round(risk_score, 1),
            overall_risk=overall_risk,
            metrics=metrics,
            alerts=alerts
        )
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_to_supabase(self, data: ShadowPortfolioData) -> Optional[Dict]:
        """Save shadow portfolio alerts to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return None
        
        record = {
            "date": data.date,
            "credit_regime": data.credit_regime,
            "vix_regime": data.vix_regime,
            "overall_risk": data.overall_risk,
            "risk_score": data.risk_score,
            "overweight_exposure": data.metrics.overweight_exposure,
            "neutral_exposure": data.metrics.neutral_exposure,
            "underweight_exposure": data.metrics.underweight_exposure,
            "max_pillar": data.metrics.max_pillar,
            "max_pillar_weight": data.metrics.max_pillar_weight,
            "alert_count": len(data.alerts),
            "alerts": [{"level": a.level, "message": a.message} for a in data.alerts],
        }
        
        try:
            response = self.supabase.table("shadow_portfolio_daily") \
                .upsert(record, on_conflict="date") \
                .execute()
            
            if response.data:
                logger.info(f"Saved shadow portfolio for {data.date}")
                return response.data[0]
        except Exception as e:
            logger.error(f"Failed to save shadow portfolio: {e}")
        
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run shadow portfolio analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Shadow Portfolio")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    calc = ShadowPortfolioCalculator()
    data = calc.calculate()
    
    print(f"\n{'='*60}")
    print(f"SHADOW PORTFOLIO - {data.date}")
    print(f"{'='*60}")
    
    print(f"\nRegimes:")
    print(f"  Credit: {data.credit_regime}")
    print(f"  VIX: {data.vix_regime}")
    print(f"  ETF Credit: {data.etf_credit_regime}")
    print(f"  Breadth 20D: {data.breadth_20d_regime}")
    print(f"  Breadth 50D: {data.breadth_50d_regime}")
    print(f"  Correlation: {data.correlation_regime}")
    
    print(f"\nRisk Assessment:")
    print(f"  Overall Risk: {data.overall_risk}")
    print(f"  Risk Score: {data.risk_score:.1f}")
    
    print(f"\nExposure Summary:")
    print(f"  Overweight: {data.metrics.overweight_exposure*100:.1f}%")
    print(f"  Neutral: {data.metrics.neutral_exposure*100:.1f}%")
    print(f"  Underweight: {data.metrics.underweight_exposure*100:.1f}%")
    print(f"  Max Concentration: {data.metrics.max_pillar} ({data.metrics.max_pillar_weight*100:.1f}%)")
    
    print(f"\nAlerts:")
    for alert in data.alerts:
        print(f"  [{alert.level}] {alert.message}")
    
    print(f"{'='*60}\n")
    
    if args.save:
        result = calc.save_to_supabase(data)
        if result:
            print("Saved to Supabase")
    
    return data


if __name__ == "__main__":
    main()
