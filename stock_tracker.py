"""
Y2AI STOCK TRACKER SERVICE
Replaces Google Sheets GOOGLEFINANCE() formulas for 18-stock portfolio

Portfolio Structure (from your Google Sheets):
- PILLAR 1: Supply Constraint (TSM, ASML, VRT)
- PILLAR 2: Capital Efficiency (GOOGL, MSFT, AMZN)
- PILLAR 3: Demand Depth (NVDA, SNOW, NOW)

Plus additional tracking stocks for complete coverage.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
import logging

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

# All tracked stocks with metadata
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
    change_today: float  # Percentage
    change_5day: float
    change_ytd: float
    retrieved_at: str


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
    
    # Metadata
    calculated_at: str
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['stocks'] = [s if isinstance(s, dict) else asdict(s) for s in self.stocks]
        result['pillars'] = [p if isinstance(p, dict) else asdict(p) for p in self.pillars]
        return result


# =============================================================================
# STOCK DATA FETCHER
# =============================================================================

class StockFetcher:
    """Fetch stock data using yfinance"""
    
    def __init__(self):
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(minutes=5)
    
    def _is_cache_valid(self) -> bool:
        if self._cache_time is None:
            return False
        return datetime.now() - self._cache_time < self._cache_duration
    
    def get_stock_data(self, ticker: str) -> Optional[Dict]:
        """Get current price and performance metrics for a ticker"""
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            
            # Get current price
            hist_1d = stock.history(period="1d")
            if hist_1d.empty:
                logger.warning(f"No data for {ticker}")
                return None
            
            current_price = float(hist_1d['Close'].iloc[-1])
            
            # Get opening price for today's change
            if len(hist_1d) > 0:
                open_price = float(hist_1d['Open'].iloc[0])
                change_today = ((current_price - open_price) / open_price) * 100
            else:
                change_today = 0.0
            
            # Get 5-day performance
            hist_5d = stock.history(period="5d")
            if len(hist_5d) >= 2:
                price_5d_ago = float(hist_5d['Close'].iloc[0])
                change_5day = ((current_price - price_5d_ago) / price_5d_ago) * 100
            else:
                change_5day = 0.0
            
            # Get YTD performance
            year_start = datetime(datetime.now().year, 1, 1)
            hist_ytd = stock.history(start=year_start)
            if len(hist_ytd) > 0:
                price_ytd_start = float(hist_ytd['Close'].iloc[0])
                change_ytd = ((current_price - price_ytd_start) / price_ytd_start) * 100
            else:
                change_ytd = 0.0
            
            return {
                "ticker": ticker,
                "price": round(current_price, 2),
                "change_today": round(change_today, 2),
                "change_5day": round(change_5day, 2),
                "change_ytd": round(change_ytd, 2)
            }
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    def get_multiple_stocks(self, tickers: List[str]) -> Dict[str, Dict]:
        """Fetch data for multiple tickers efficiently"""
        results = {}
        
        try:
            import yfinance as yf
            
            # Batch download for efficiency
            tickers_str = " ".join(tickers)
            data = yf.download(tickers_str, period="5d", progress=False)
            
            # Also get YTD data
            year_start = datetime(datetime.now().year, 1, 1)
            data_ytd = yf.download(tickers_str, start=year_start, progress=False)
            
            for ticker in tickers:
                try:
                    if len(tickers) > 1:
                        close = data['Close'][ticker]
                        close_ytd = data_ytd['Close'][ticker]
                    else:
                        close = data['Close']
                        close_ytd = data_ytd['Close']
                    
                    if close.empty:
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
                    if not close_ytd.empty:
                        price_ytd = float(close_ytd.iloc[0])
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
                    logger.error(f"Error processing {ticker}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Batch download error: {e}")
            # Fallback to individual fetches
            for ticker in tickers:
                data = self.get_stock_data(ticker)
                if data:
                    results[ticker] = data
        
        return results


# =============================================================================
# STOCK TRACKER
# =============================================================================

class StockTracker:
    """
    Main stock tracker - replaces your Google Sheets stock tracking
    
    Calculates:
    - Individual stock performance (today, 5-day, YTD)
    - Pillar averages (Supply, Capital, Demand)
    - Y2AI Index (equal-weight of 9 pillar stocks)
    - Comparison vs SPY, QQQ, Mag7
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
        Determine daily signal status
        
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
        Generate complete daily stock tracker report
        
        This replaces your Google Sheets Dashboard tab.
        """
        logger.info("Generating Y2AI Stock Tracker Report...")
        
        # Get all tickers
        all_tickers = list(STOCK_METADATA.keys()) + list(BENCHMARKS.keys())
        
        # Fetch stock data
        stock_data = self.fetcher.get_multiple_stocks(all_tickers)
        
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
                    retrieved_at=datetime.utcnow().isoformat()
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
        
        # Find best/worst stocks (today)
        sorted_stocks = sorted(stocks, key=lambda s: s.change_today, reverse=True)
        best_stock = sorted_stocks[0].ticker if sorted_stocks else "N/A"
        worst_stock = sorted_stocks[-1].ticker if sorted_stocks else "N/A"
        
        # Find best/worst pillars (today)
        sorted_pillars = sorted(pillars, key=lambda p: p.avg_today, reverse=True)
        best_pillar = sorted_pillars[0].name if sorted_pillars else "N/A"
        worst_pillar = sorted_pillars[-1].name if sorted_pillars else "N/A"
        
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
            calculated_at=datetime.utcnow().isoformat()
        )
        
        logger.info(f"  Y2AI Index Today: {report.y2ai_index_today:+.2f}%")
        logger.info(f"  SPY Today: {report.spy_today:+.2f}%")
        logger.info(f"  Status: {report.status}")
        
        return report
    
    def format_for_social(self, report: DailyReport) -> str:
        """
        Format report for social media posting
        
        Template from your Google Sheets daily X posting strategy.
        """
        emoji = {
            "VALIDATING": "✅",
            "NEUTRAL": "⚪",
            "CONTRADICTING": "⚠️"
        }.get(report.status, "⚪")
        
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
        
        post = f"""📊 Y2AI Infrastructure Index | {report.date}

Index: {report.y2ai_index_today:+.2f}%
S&P 500: {report.spy_today:+.2f}%

{emoji} {status_text}

Best: {report.best_stock} ({next((s.change_today for s in report.stocks if s.ticker == report.best_stock), 0):+.1f}%)
Worst: {report.worst_stock} ({next((s.change_today for s in report.stocks if s.ticker == report.worst_stock), 0):+.1f}%)

YTD: Y2AI {report.y2ai_index_ytd:+.1f}% | SPY {report.spy_ytd:+.1f}%

#AI #Infrastructure #MarketData"""
        
        return post


# =============================================================================
# STORAGE INTEGRATION
# =============================================================================

def store_report(report: DailyReport):
    """Store daily report in Supabase"""
    try:
        from .storage import get_storage
        
        storage = get_storage()
        if hasattr(storage, 'client') and storage.is_connected():
            storage.client.table("stock_tracker_daily").upsert(
                report.to_dict(),
                on_conflict="date"
            ).execute()
            logger.info(f"Stored report for {report.date}")
    except Exception as e:
        logger.error(f"Storage error: {e}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    tracker = StockTracker()
    report = tracker.generate_daily_report()
    
    print(f"\n{'='*60}")
    print("Y2AI STOCK TRACKER REPORT")
    print(f"{'='*60}")
    print(f"Date: {report.date}")
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
