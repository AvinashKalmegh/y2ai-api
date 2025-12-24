"""
Market Events Calendar

Tracks upcoming market-moving events (FOMC, CPI, earnings, etc.)
and their potential impact on regime.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketEvent:
    """A market-moving event"""
    name: str
    date: str
    event_type: str  # FOMC, CPI, JOBS, EARNINGS, OPEX
    impact: str      # HIGH, MEDIUM, LOW
    description: str = ""


@dataclass
class EventsResult:
    """Events analysis result"""
    next_event: Optional[MarketEvent]
    days_to_event: int
    event_status: str  # CLEAR, APPROACHING, IMMINENT
    upcoming_events: List[MarketEvent]
    display: str
    
    def to_dict(self) -> dict:
        result = {
            "next_event": asdict(self.next_event) if self.next_event else None,
            "days_to_event": self.days_to_event,
            "event_status": self.event_status,
            "upcoming_events": [asdict(e) for e in self.upcoming_events],
            "display": self.display,
        }
        return result


class EventsTracker:
    """
    Track market-moving events.
    
    Key events:
    - FOMC meetings (8x/year)
    - CPI releases (monthly)
    - Jobs reports (monthly first Friday)
    - NVDA earnings (quarterly)
    - Options expiration (quad witching)
    """
    
    def __init__(self):
        self.events = self._build_calendar()
    
    def _build_calendar(self) -> List[MarketEvent]:
        """Build calendar of known events for 2024-2025"""
        events = []
        
        # 2025 FOMC meetings (approximate)
        fomc_dates = [
            "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
            "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17"
        ]
        for date in fomc_dates:
            events.append(MarketEvent(
                name="FOMC Decision",
                date=date,
                event_type="FOMC",
                impact="HIGH",
                description="Federal Reserve interest rate decision"
            ))
        
        # 2025 CPI releases (approximate - usually mid-month)
        cpi_dates = [
            "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
            "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-13",
            "2025-09-11", "2025-10-10", "2025-11-13", "2025-12-11"
        ]
        for date in cpi_dates:
            events.append(MarketEvent(
                name="CPI Report",
                date=date,
                event_type="CPI",
                impact="HIGH",
                description="Consumer Price Index release"
            ))
        
        # Jobs reports (first Friday of each month)
        jobs_dates = [
            "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
            "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
            "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05"
        ]
        for date in jobs_dates:
            events.append(MarketEvent(
                name="Jobs Report",
                date=date,
                event_type="JOBS",
                impact="HIGH",
                description="Non-farm payrolls"
            ))
        
        # NVDA earnings (approximate)
        nvda_dates = ["2025-02-26", "2025-05-28", "2025-08-27", "2025-11-19"]
        for date in nvda_dates:
            events.append(MarketEvent(
                name="NVDA Earnings",
                date=date,
                event_type="EARNINGS",
                impact="HIGH",
                description="NVIDIA quarterly earnings"
            ))
        
        # Quad witching (3rd Friday of March, June, Sept, Dec)
        opex_dates = ["2025-03-21", "2025-06-20", "2025-09-19", "2025-12-19"]
        for date in opex_dates:
            events.append(MarketEvent(
                name="Quad Witching",
                date=date,
                event_type="OPEX",
                impact="MEDIUM",
                description="Options expiration"
            ))
        
        # Sort by date
        events.sort(key=lambda x: x.date)
        return events
    
    def get_upcoming_events(self, days: int = 14) -> List[MarketEvent]:
        """Get events in the next N days"""
        today = datetime.now().date()
        cutoff = today + timedelta(days=days)
        
        upcoming = []
        for event in self.events:
            event_date = datetime.strptime(event.date, "%Y-%m-%d").date()
            if today <= event_date <= cutoff:
                upcoming.append(event)
        
        return upcoming
    
    def get_next_event(self) -> Optional[MarketEvent]:
        """Get the very next event"""
        today = datetime.now().date()
        
        for event in self.events:
            event_date = datetime.strptime(event.date, "%Y-%m-%d").date()
            if event_date >= today:
                return event
        
        return None
    
    def days_until(self, event: MarketEvent) -> int:
        """Calculate days until an event"""
        today = datetime.now().date()
        event_date = datetime.strptime(event.date, "%Y-%m-%d").date()
        return (event_date - today).days
    
    def calculate(self) -> EventsResult:
        """Calculate events analysis"""
        next_event = self.get_next_event()
        upcoming = self.get_upcoming_events(14)
        
        if not next_event:
            return EventsResult(
                next_event=None,
                days_to_event=999,
                event_status="CLEAR",
                upcoming_events=[],
                display="No events scheduled",
            )
        
        days = self.days_until(next_event)
        
        if days <= 2:
            status = "IMMINENT"
            emoji = "⚡"
        elif days <= 7:
            status = "APPROACHING"
            emoji = "📅"
        else:
            status = "CLEAR"
            emoji = "✓"
        
        display = f"{emoji} {next_event.name} in {days}D"
        
        return EventsResult(
            next_event=next_event,
            days_to_event=days,
            event_status=status,
            upcoming_events=upcoming,
            display=display,
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    tracker = EventsTracker()
    result = tracker.calculate()
    
    print(f"\n{'='*60}")
    print("EVENTS CALENDAR")
    print(f"{'='*60}")
    print(f"Next Event: {result.next_event.name if result.next_event else 'None'}")
    print(f"Days Away: {result.days_to_event}")
    print(f"Status: {result.event_status}")
    print(f"Display: {result.display}")
    print(f"\nUpcoming (14 days):")
    for event in result.upcoming_events:
        print(f"  - {event.date}: {event.name} ({event.impact})")
