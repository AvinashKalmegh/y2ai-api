"""
Y2AI STOCK TRACKER SERVICE (ENHANCED)
With robust error handling, batch fetching, and health tracking

Portfolio Structure:
- PILLAR 1: Supply Constraint (TSM, ASML, VRT)
- PILLAR 2: Capital Efficiency (GOOGL, MSFT, AMZN)
- PILLAR 3: Demand Depth (NVDA, SNOW, NOW)

Plus additional tracking stocks for complete coverage.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field

import pandas as pd

# Import resilience module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.resilience import (
    resilient_call,
    with_fallback,
    get_health_tracker,
    RetryExhaustedError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PORTFOLIO CONFIGURATION
# =============================================================================

PILLARS = {
    "supply_constraint": {
        "name": "Supply Constraint",
        "description": "Chip manufacturing and infrastructure bottlenecks",
        "stocks": ["TSM", "ASML", "VRT"]
    },
    "capital_efficiency": {
        "name": "Capital Efficiency",
        "description": "Self-funded AI capex, contracted demand",
        "stocks": ["GOOGL", "MSFT", "AMZN"]
    },
    "demand_depth": {
        "name": "Demand Depth",
        "description": "GPU pricing power, enterprise adoption",
        "stocks": ["NVDA", "SNOW", "NOW"]
    }
}

STOCK_METADATA = {
    # Supply Constraint Pillar
    "TSM": {"name": "Taiwan Semiconductor", "pillar": "supply_constraint", 
            "thesis": "100% utilization through 2028"},
    "ASML": {"name": "ASML Holding", "pillar": "supply_constraint",
             "thesis": "Only EUV lithography supplier"},
    "VRT": {"name": "Vertiv Holdings", "pillar": "supply_constraint",
            "thesis": "AI data centers need 3-5x power density"},
    
    # Capital Efficiency Pillar
    "GOOGL": {"name": "Alphabet/Google", "pillar": "capital_efficiency",
              "thesis": "$24.5B FCF covers $23.9B capex"},
    "MSFT": {"name": "Microsoft", "pillar": "capital_efficiency",
             "thesis": "$392B RPO backlog"},
    "AMZN": {"name": "Amazon", "pillar": "capital_efficiency",
             "thesis": "69% FCF decline accepted for AI leadership"},
    
    # Demand Depth Pillar
    "NVDA": {"name": "NVIDIA", "pillar": "demand_depth",
             "thesis": "12-month sold out, pricing power"},
    "SNOW": {"name": "Snowflake", "pillar": "demand_depth",
             "thesis": "Enterprise adoption beyond hyperscalers"},
    "NOW": {"name": "ServiceNow", "pillar": "demand_depth",
            "thesis": "AI embedded in workflow operations"},
    
    # Additional Coverage
    "META": {"name": "Meta Platforms", "pillar": None,
             "thesis": "$70B capex commitment 2025"},
    "AMD": {"name": "AMD", "pillar": None,
            "thesis": "NVIDIA alternative, MI300 ramp"},
    "INTC": {"name": "Intel", "pillar": None,
             "thesis": "Foundry strategy execution"},
    "AVGO": {"name": "Broadcom", "pillar": None,
             "thesis": "Custom AI chips, VMware integration"},
    "ORCL": {"name": "Oracle", "pillar": None,
             "thesis": "Cloud + AI database"},
    "CRM": {"name": "Salesforce", "pillar": None,
            "thesis": "Enterprise AI adoption leader"},
    "EQIX": {"name": "Equinix", "pillar": None,
             "thesis": "Data center REIT, AI infrastructure"},
    "DLR": {"name": "Digital Realty", "pillar": None,
            "thesis": "Data center REIT"},
    "PLTR": {"name": "Palantir", "pillar": None,
             "thesis": "Government + Enterprise AI"},
}

BENCHMARKS = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class StockReading:
    """Individual stock reading"""
    ticker: str
    name: str
    pillar: Optional[str]
    price: float
    change_today: float
    change_5day: float
    change_ytd: float
    retrieved_at: str
    data_source: str = "live"  # "live" or "failed"


@dataclass
class PillarPerformance:
    """Pillar-level aggregated performance"""
    pillar_id: str
    name: str
    stocks: List[str]
    avg_today: float
    avg_5day: float
    avg_ytd: float


@dataclass
class DailyReport:
    """Complete daily stock tracker report"""
    date: str
    
    # Individual stocks
    stocks: List[StockReading]
    
    # Pillar performance
    pillars: List[PillarPerformance]
    
    # Y2AI Index (equal-weight of 9 pillar stocks)
    y2ai_index_today: float
    y2ai_index_5day: float
    y2ai_index_ytd: float
    
    # Benchmarks
    spy_today: float
    spy_5day: float
    spy_ytd: float
    qqq_today: float
    qqq_5day: float
    qqq_ytd: float
    
    # Signals
    status: str  # VALIDATING, NEUTRAL, CONTRADICTING
    best_stock: str
    worst_stock: str
    best_pillar: str
    worst_pillar: str
    
    # Data quality
    stocks_fetched: int = 0
    stocks_failed: int = 0
    
    # Metadata
    calculated_at: str = ""
    
    def __post_init__(self):
        if not self.calculated_at:
            self.calculated_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['stocks'] = [s if isinstance(s, dict) else asdict(s) for s in self.stocks]
        result['pillars'] = [p if isinstance(p, dict) else asdict(p) for p in self.pillars]
        return result
    
    @property
    def data_quality_score(self) -> float:
        """Score from 0-1 indicating how much live data we have"""
        total = self.stocks_fetched + self.stocks_failed
        if total == 0:
            return 1.0
        return self.stocks_fetched / total


# =============================================================================
# STOCK DATA FETCHER (Enhanced)
# =============================================================================

class StockFetcher:
    """Fetch stock data with resilience and batch optimization"""
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=5)
        self._failed_tickers: Dict[str, datetime] = {}
        self._ticker_cooldown = timedelta(minutes=30)
    
    def _is_cache_valid(self) -> bool:
        if self._cache_time is None:
            return False
        return datetime.utcnow() - self._cache_time < self._cache_duration
    
    def _is_ticker_available(self, ticker: str) -> bool:
        """Check if ticker is not in cooldown from recent failures"""
        if ticker in self._failed_tickers:
            if datetime.utcnow() - self._failed_tickers[ticker] < self._ticker_cooldown:
                return False
            del self._failed_tickers[ticker]
        return True
    
    def _mark_ticker_failed(self, ticker: str):
        """Mark a ticker as failed"""
        self._failed_tickers[ticker] = datetime.utcnow()
    
    @resilient_call(
        service_name="yfinance_stocks",
        max_retries=3,
        base_delay=1.0,
        use_circuit_breaker=True,
        use_rate_limiter=False,
    )
    def _batch_download(self, tickers: List[str], period: str = "5d") -> pd.DataFrame:
        """Batch download stock data with resilience"""
        import yfinance as yf
        tickers_str = " ".join(tickers)
        return yf.download(tickers_str, period=period, progress=False, threads=True)
    
    @resilient_call(
        service_name="yfinance_stocks",
        max_retries=2,
        base_delay=1.0,
        use_circuit_breaker=True,
        use_rate_limiter=False,
    )
    def _fetch_ytd_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch YTD data for stocks"""
        import yfinance as yf
        year_start = datetime(datetime.now().year, 1, 1)
        tickers_str = " ".join(tickers)
        return yf.download(tickers_str, start=year_start, progress=False, threads=True)
    
    def get_multiple_stocks(self, tickers: List[str]) -> Tuple[Dict[str, Dict], List[str]]:
        """
        Fetch data for multiple tickers efficiently.
        
        Returns:
            Tuple of (results dict, failed tickers list)
        """
        # Check cache
        if self._is_cache_valid():
            cached_results = {t: self._cache[t] for t in tickers if t in self._cache}
            if len(cached_results) == len(tickers):
                return cached_results, []
        
        # Filter out tickers in cooldown
        available_tickers = [t for t in tickers if self._is_ticker_available(t)]
        cooldown_tickers = [t for t in tickers if t not in available_tickers]
        
        if cooldown_tickers:
            logger.debug(f"Tickers in cooldown: {cooldown_tickers}")
        
        results = {}
        failed = list(cooldown_tickers)
        
        if not available_tickers:
            return results, failed
        
        try:
            # Batch download 5-day data
            data = self._batch_download(available_tickers, "5d")
            
            # Fetch YTD data
            data_ytd = self._fetch_ytd_data(available_tickers)
            
            for ticker in available_tickers:
                try:
                    # Handle single vs multiple ticker response format
                    if len(available_tickers) > 1:
                        close = data['Close'][ticker] if ticker in data['Close'].columns else None
                        close_ytd = data_ytd['Close'][ticker] if ticker in data_ytd['Close'].columns else None
                    else:
                        close = data['Close']
                        close_ytd = data_ytd['Close']
                    
                    if close is None or close.empty:
                        self._mark_ticker_failed(ticker)
                        failed.append(ticker)
                        continue
                    
                    # Remove NaN values
                    close = close.dropna()
                    if close.empty:
                        self._mark_ticker_failed(ticker)
                        failed.append(ticker)
                        continue
                    
                    current_price = float(close.iloc[-1])
                    
                    # Today's change (last vs previous close)
                    if len(close) >= 2:
                        prev_close = float(close.iloc[-2])
                        change_today = ((current_price - prev_close) / prev_close) * 100
                    else:
                        change_today = 0.0
                    
                    # 5-day change
                    if len(close) >= 5:
                        price_5d = float(close.iloc[0])
                        change_5day = ((current_price - price_5d) / price_5d) * 100
                    else:
                        change_5day = 0.0
                    
                    # YTD change
                    if close_ytd is not None and not close_ytd.dropna().empty:
                        price_ytd = float(close_ytd.dropna().iloc[0])
                        change_ytd = ((current_price - price_ytd) / price_ytd) * 100
                    else:
                        change_ytd = 0.0
                    
                    results[ticker] = {
                        "ticker": ticker,
                        "price": round(current_price, 2),
                        "change_today": round(change_today, 2),
                        "change_5day": round(change_5day, 2),
                        "change_ytd": round(change_ytd, 2)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing {ticker}: {e}")
                    self._mark_ticker_failed(ticker)
                    failed.append(ticker)
            
            # Update cache
            self._cache.update(results)
            self._cache_time = datetime.utcnow()
            
        except RetryExhaustedError as e:
            logger.error(f"Batch download failed after retries: {e}")
            failed.extend(available_tickers)
        except Exception as e:
            logger.error(f"Unexpected batch download error: {e}")
            failed.extend(available_tickers)
        
        logger.info(f"Stock fetch: {len(results)} success, {len(failed)} failed")
        return results, failed
    
    def get_health(self) -> Dict[str, Any]:
        """Get fetcher health status"""
        tracker = get_health_tracker("yfinance_stocks")
        return {
            **tracker.to_dict(),
            "cached_tickers": len(self._cache),
            "failed_tickers_in_cooldown": list(self._failed_tickers.keys()),
        }


# =============================================================================
# STOCK TRACKER (Enhanced)
# =============================================================================

class StockTracker:
    """
    Main stock tracker with resilience and graceful degradation.
    
    Calculates:
    - Individual stock performance (today, 5-day, YTD)
    - Pillar averages (Supply, Capital, Demand)
    - Y2AI Index (equal-weight of 9 pillar stocks)
    - Comparison vs SPY, QQQ
    - Daily signal (VALIDATING/NEUTRAL/CONTRADICTING)
    """
    
    def __init__(self):
        self.fetcher = StockFetcher()
    
    def _calculate_pillar_performance(
        self, 
        stock_data: Dict[str, Dict],
        pillar_id: str
    ) -> Optional[PillarPerformance]:
        """Calculate average performance for a pillar"""
        pillar = PILLARS.get(pillar_id)
        if not pillar:
            return None
        
        stocks = pillar["stocks"]
        today_values = []
        day5_values = []
        ytd_values = []
        
        for ticker in stocks:
            if ticker in stock_data:
                data = stock_data[ticker]
                today_values.append(data["change_today"])
                day5_values.append(data["change_5day"])
                ytd_values.append(data["change_ytd"])
        
        if not today_values:
            return None
        
        return PillarPerformance(
            pillar_id=pillar_id,
            name=pillar["name"],
            stocks=stocks,
            avg_today=round(sum(today_values) / len(today_values), 2),
            avg_5day=round(sum(day5_values) / len(day5_values), 2),
            avg_ytd=round(sum(ytd_values) / len(ytd_values), 2)
        )
    
    def _calculate_y2ai_index(self, stock_data: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate Y2AI Index (equal-weight of 9 pillar stocks)"""
        pillar_stocks = []
        for pillar in PILLARS.values():
            pillar_stocks.extend(pillar["stocks"])
        
        today_values = []
        day5_values = []
        ytd_values = []
        
        for ticker in pillar_stocks:
            if ticker in stock_data:
                data = stock_data[ticker]
                today_values.append(data["change_today"])
                day5_values.append(data["change_5day"])
                ytd_values.append(data["change_ytd"])
        
        return {
            "today": round(sum(today_values) / len(today_values), 2) if today_values else 0,
            "5day": round(sum(day5_values) / len(day5_values), 2) if day5_values else 0,
            "ytd": round(sum(ytd_values) / len(ytd_values), 2) if ytd_values else 0
        }
    
    def _determine_status(self, y2ai_today: float, spy_today: float) -> str:
        """
        Determine daily signal status.
        
        VALIDATING: Y2AI outperforms SPY by >0.25%
        NEUTRAL: Y2AI within ±0.25% of SPY
        CONTRADICTING: Y2AI underperforms SPY by >0.25%
        """
        diff = y2ai_today - spy_today
        
        if diff > 0.25:
            return "VALIDATING"
        elif diff < -0.25:
            return "CONTRADICTING"
        else:
            return "NEUTRAL"
    
    def generate_daily_report(self) -> DailyReport:
        """
        Generate complete daily stock tracker report.
        
        Returns a report even if some stocks fail to fetch.
        """
        logger.info("Generating Y2AI Stock Tracker Report...")
        
        # Get all tickers
        all_tickers = list(STOCK_METADATA.keys()) + list(BENCHMARKS.keys())
        
        # Fetch stock data
        stock_data, failed_tickers = self.fetcher.get_multiple_stocks(all_tickers)
        
        # Build stock readings
        stocks = []
        for ticker, meta in STOCK_METADATA.items():
            if ticker in stock_data:
                data = stock_data[ticker]
                stocks.append(StockReading(
                    ticker=ticker,
                    name=meta["name"],
                    pillar=meta["pillar"],
                    price=data["price"],
                    change_today=data["change_today"],
                    change_5day=data["change_5day"],
                    change_ytd=data["change_ytd"],
                    retrieved_at=datetime.utcnow().isoformat(),
                    data_source="live"
                ))
            elif ticker in failed_tickers:
                # Include failed stocks with zero values
                stocks.append(StockReading(
                    ticker=ticker,
                    name=meta["name"],
                    pillar=meta["pillar"],
                    price=0.0,
                    change_today=0.0,
                    change_5day=0.0,
                    change_ytd=0.0,
                    retrieved_at=datetime.utcnow().isoformat(),
                    data_source="failed"
                ))
        
        # Calculate pillar performance
        pillars = []
        for pillar_id in PILLARS.keys():
            perf = self._calculate_pillar_performance(stock_data, pillar_id)
            if perf:
                pillars.append(perf)
        
        # Calculate Y2AI Index
        y2ai = self._calculate_y2ai_index(stock_data)
        
        # Get benchmark data
        spy = stock_data.get("SPY", {"change_today": 0, "change_5day": 0, "change_ytd": 0})
        qqq = stock_data.get("QQQ", {"change_today": 0, "change_5day": 0, "change_ytd": 0})
        
        # Determine status
        status = self._determine_status(y2ai["today"], spy["change_today"])
        
        # Find best/worst stocks (from successfully fetched only)
        live_stocks = [s for s in stocks if s.data_source == "live"]
        if live_stocks:
            sorted_stocks = sorted(live_stocks, key=lambda s: s.change_today, reverse=True)
            best_stock = sorted_stocks[0].ticker
            worst_stock = sorted_stocks[-1].ticker
        else:
            best_stock = "N/A"
            worst_stock = "N/A"
        
        # Find best/worst pillars
        if pillars:
            sorted_pillars = sorted(pillars, key=lambda p: p.avg_today, reverse=True)
            best_pillar = sorted_pillars[0].name
            worst_pillar = sorted_pillars[-1].name
        else:
            best_pillar = "N/A"
            worst_pillar = "N/A"
        
        report = DailyReport(
            date=datetime.now().strftime("%Y-%m-%d"),
            stocks=stocks,
            pillars=pillars,
            y2ai_index_today=y2ai["today"],
            y2ai_index_5day=y2ai["5day"],
            y2ai_index_ytd=y2ai["ytd"],
            spy_today=spy["change_today"],
            spy_5day=spy["change_5day"],
            spy_ytd=spy["change_ytd"],
            qqq_today=qqq["change_today"],
            qqq_5day=qqq["change_5day"],
            qqq_ytd=qqq["change_ytd"],
            status=status,
            best_stock=best_stock,
            worst_stock=worst_stock,
            best_pillar=best_pillar,
            worst_pillar=worst_pillar,
            stocks_fetched=len(stock_data),
            stocks_failed=len(failed_tickers),
            calculated_at=datetime.utcnow().isoformat()
        )
        
        logger.info(f"  Y2AI Index Today: {report.y2ai_index_today:+.2f}%")
        logger.info(f"  SPY Today: {report.spy_today:+.2f}%")
        logger.info(f"  Status: {report.status}")
        logger.info(f"  Data Quality: {report.data_quality_score:.0%}")
        
        return report
    
    def format_for_social(self, report: DailyReport) -> str:
        """Format report for social media posting"""
        emoji = {
            "VALIDATING": "✓",
            "NEUTRAL": "○",
            "CONTRADICTING": "⚠"
        }.get(report.status, "○")
        
        # Determine if strongly validating/contradicting
        diff = report.y2ai_index_today - report.spy_today
        if diff > 1.0:
            emoji = "🔥"
            status_text = "STRONGLY VALIDATING"
        elif diff < -1.0:
            emoji = "📉"
            status_text = "STRONGLY CONTRADICTING"
        else:
            status_text = report.status
        
        # Find best/worst stock changes
        live_stocks = [s for s in report.stocks if s.data_source == "live"]
        best_change = next((s.change_today for s in live_stocks if s.ticker == report.best_stock), 0)
        worst_change = next((s.change_today for s in live_stocks if s.ticker == report.worst_stock), 0)
        
        post = f"""📊 Y2AI Infrastructure Index | {report.date}

Index: {report.y2ai_index_today:+.2f}%
S&P 500: {report.spy_today:+.2f}%

{emoji} {status_text}

Best: {report.best_stock} ({best_change:+.1f}%)
Worst: {report.worst_stock} ({worst_change:+.1f}%)

YTD: Y2AI {report.y2ai_index_ytd:+.1f}% | SPY {report.spy_ytd:+.1f}%

Dashboard: https://y2ai.us

#AI #Infrastructure #MarketData"""
        
        return post
    
    def get_health(self) -> Dict[str, Any]:
        """Get tracker health status"""
        return self.fetcher.get_health()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    tracker = StockTracker()
    
    # Show health status first
    print(f"\n{'='*60}")
    print("STOCK FETCHER HEALTH")
    print(f"{'='*60}")
    health = tracker.get_health()
    print(f"Success rate: {health.get('success_rate', 100):.1f}%")
    print(f"Cached tickers: {health.get('cached_tickers', 0)}")
    if health.get('failed_tickers_in_cooldown'):
        print(f"In cooldown: {health['failed_tickers_in_cooldown']}")
    
    # Generate report
    print(f"\n{'='*60}")
    print("GENERATING REPORT")
    print(f"{'='*60}")
    report = tracker.generate_daily_report()
    
    print(f"\n{'='*60}")
    print("Y2AI STOCK TRACKER REPORT (ENHANCED)")
    print(f"{'='*60}")
    print(f"Date: {report.date}")
    print(f"Data Quality: {report.data_quality_score:.0%}")
    print(f"Status: {report.status}")
    print()
    
    print("PILLAR PERFORMANCE:")
    for p in report.pillars:
        print(f"  {p.name}: {p.avg_today:+.2f}% today | {p.avg_ytd:+.1f}% YTD")
    print()
    
    print("Y2AI INDEX:")
    print(f"  Today:  {report.y2ai_index_today:+.2f}%")
    print(f"  5-Day:  {report.y2ai_index_5day:+.2f}%")
    print(f"  YTD:    {report.y2ai_index_ytd:+.1f}%")
    print()
    
    print("VS BENCHMARKS:")
    print(f"  SPY:    {report.spy_today:+.2f}% today | {report.spy_ytd:+.1f}% YTD")
    print(f"  QQQ:    {report.qqq_today:+.2f}% today | {report.qqq_ytd:+.1f}% YTD")
    print()
    
    print("BEST/WORST:")
    print(f"  Best Stock:  {report.best_stock}")
    print(f"  Worst Stock: {report.worst_stock}")
    print(f"  Best Pillar: {report.best_pillar}")
    print()
    
    print("SOCIAL POST:")
    print("-" * 40)
    print(tracker.format_for_social(report))
