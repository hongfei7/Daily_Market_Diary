from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class WatchlistDefinition:
    ticker: str
    name: str
    sector: str
    bucket: str
    thesis: str = ""
    upcoming_catalyst: str = ""
    catalyst_date: str = ""


@dataclass
class WatchlistSnapshot:
    definition: WatchlistDefinition
    last_price: Optional[float] = None
    daily_change_pct: Optional[float] = None
    range_position_pct: Optional[float] = None
    range_label: str = ""
    note: str = ""
    recent_news: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["ticker"] = self.definition.ticker
        payload["name"] = self.definition.name
        payload["sector"] = self.definition.sector
        payload["bucket"] = self.definition.bucket
        payload["thesis"] = self.definition.thesis
        payload["upcoming_catalyst"] = self.definition.upcoming_catalyst
        payload["catalyst_date"] = self.definition.catalyst_date
        return payload
