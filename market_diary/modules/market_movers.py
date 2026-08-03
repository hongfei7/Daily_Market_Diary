"""Market movers, ETF flow proxies, and options activity adapters."""

import os
from typing import Any, Dict, List, Optional

import yfinance as yf

from market_diary.modules.adapter_shortsell import fetch_short_sell_data
from market_diary.modules.provenance import provenance_record, unavailable_record

YFINANCE_TIMEOUT = float(os.environ.get("DMD_YFINANCE_TIMEOUT_SECONDS", "6"))


class MarketMoversAnalyzer:
    """Collect public ETF activity proxies and official short-selling data."""

    MAJOR_ETFS = {
        "SPY": "S&P 500 ETF",
        "QQQ": "Nasdaq 100 ETF",
        "FXI": "China Large-Cap ETF",
        "KWEB": "China Internet ETF",
        "2800.HK": "Tracker Fund of Hong Kong",
        "2828.HK": "HSCEI ETF",
        "3033.HK": "Hang Seng TECH ETF",
        "TLT": "20+ Year Treasury ETF",
        "HYG": "High Yield Bond ETF",
        "LQD": "Investment Grade Bond ETF",
        "GLD": "Gold ETF",
        "USO": "Oil ETF",
    }

    def fetch_premarket_movers(self, top_n: int = 10) -> Dict:
        """Return no mover claims until a dated, attributable feed is configured."""
        return {
            "gainers": [],
            "losers": [],
            "most_active": [],
        }

    def _fetch_top_gainers(self, n: int) -> List[Dict]:
        return []

    def _fetch_top_losers(self, n: int) -> List[Dict]:
        return []

    def _fetch_most_active(self, n: int) -> List[Dict]:
        return []

    def fetch_etf_flows(self, date: str) -> List[Dict]:
        """Approximate ETF flow pressure using volume and daily price change."""
        flows: List[Dict] = []

        for ticker, name in self.MAJOR_ETFS.items():
            try:
                hist = yf.Ticker(ticker).history(period="5d", timeout=YFINANCE_TIMEOUT)
                if hist.empty or len(hist) < 2:
                    continue

                recent_volume = hist["Volume"].iloc[-1]
                avg_volume = hist["Volume"].mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                price_change_pct = ((hist["Close"].iloc[-1] / hist["Close"].iloc[-2]) - 1) * 100
                estimated_flow = recent_volume * price_change_pct / 100

                flows.append(
                    {
                        "ticker": ticker,
                        "name": name,
                        "price": round(float(hist["Close"].iloc[-1]), 2),
                        "change_pct": round(float(price_change_pct), 2),
                        "volume": int(recent_volume),
                        "volume_ratio": round(float(volume_ratio), 2),
                        "estimated_flow_direction": "inflow" if estimated_flow > 0 else "outflow",
                        "as_of": str(hist.index[-1].date()),
                        "source": "Yahoo Finance",
                        "source_url": f"https://finance.yahoo.com/quote/{ticker}/",
                    }
                )
            except Exception as exc:
                print(f"[market_movers] Error fetching ETF {ticker}: {exc}")

        flows.sort(key=lambda item: abs(item["volume_ratio"] - 1.0), reverse=True)
        return flows[:10]

    def fetch_block_trades_cn(self, date: str) -> List[Dict]:
        """Return no block-trade claims until an attributable feed is configured."""
        return []

    def fetch_unusual_options(self, date: str) -> List[Dict]:
        """Return no unusual-options claims until an attributable feed is configured."""
        return []

    def format_for_report(
        self,
        movers: Dict,
        etf_flows: List[Dict],
        block_trades: List[Dict],
        options: List[Dict],
    ) -> str:
        """Format the mover payload into a readable fallback block."""
        lines: List[str] = ["### Movers and Flow Proxies", ""]

        if movers.get("gainers"):
            lines.append("#### Premarket Gainers")
            lines.append("")
            lines.append("| Ticker | Company | Price | Change | Volume | Catalyst |")
            lines.append("|--------|---------|-------|--------|--------|----------|")
            for stock in movers["gainers"][:5]:
                lines.append(
                    f"| {stock['ticker']} | {stock['company']} | ${stock['price']:.2f} | "
                    f"{stock['change_pct']:+.2f}% | {stock['volume']:,} | {stock['catalyst']} |"
                )
            lines.append("")

        if movers.get("losers"):
            lines.append("#### Premarket Losers")
            lines.append("")
            lines.append("| Ticker | Company | Price | Change | Volume | Catalyst |")
            lines.append("|--------|---------|-------|--------|--------|----------|")
            for stock in movers["losers"][:5]:
                lines.append(
                    f"| {stock['ticker']} | {stock['company']} | ${stock['price']:.2f} | "
                    f"{stock['change_pct']:+.2f}% | {stock['volume']:,} | {stock['catalyst']} |"
                )
            lines.append("")

        if etf_flows:
            lines.append("#### ETF Flow Proxies")
            lines.append("")
            lines.append("| ETF | Name | Price | Change | Volume Ratio | Direction |")
            lines.append("|-----|------|-------|--------|--------------|-----------|")
            for etf in etf_flows[:8]:
                lines.append(
                    f"| {etf['ticker']} | {etf['name']} | ${etf['price']:.2f} | "
                    f"{etf['change_pct']:+.2f}% | {etf['volume_ratio']:.2f}x | "
                    f"{etf['estimated_flow_direction']} |"
                )
            lines.append("")

        if options:
            lines.append("#### Unusual Options")
            lines.append("")
            for opt in options[:5]:
                lines.append(
                    f"- **{opt['ticker']}** {opt['option_type']} {opt['strike']} expiring {opt['expiry']} | "
                    f"Volume {opt['volume']:,} | Vol/OI {opt['volume_oi_ratio']:.2f}x | "
                    f"IV {opt['implied_vol']:.1f}% | {opt['sentiment']}"
                )
            lines.append("")

        if block_trades:
            lines.append("#### Greater China Block Trades")
            lines.append("")
            lines.append("| Ticker | Name | Volume | Price | Discount/Premium | Buyer | Seller |")
            lines.append("|--------|------|--------|-------|------------------|-------|--------|")
            for trade in block_trades[:5]:
                lines.append(
                    f"| {trade['ticker']} | {trade['name']} | {trade['volume']:,} | "
                    f"{trade['price']:.2f} | {trade['premium']:+.2f}% | "
                    f"{trade['buyer']} | {trade['seller']} |"
                )
            lines.append("")

        return "\n".join(lines)


def fetch_movers_data(date: str, watchlists: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict:
    """Public entry point for market movers and flow data."""
    analyzer = MarketMoversAnalyzer()
    movers = analyzer.fetch_premarket_movers(top_n=10)
    etf_flows = analyzer.fetch_etf_flows(date)
    block_trades = analyzer.fetch_block_trades_cn(date)
    options = analyzer.fetch_unusual_options(date)
    short_sell = fetch_short_sell_data(date, watchlists=watchlists)

    provenance = []
    if etf_flows:
        provenance.append(
            provenance_record(
                source_name="Yahoo Finance market quotes",
                source_url="https://finance.yahoo.com/",
                as_of=max(str(item.get("as_of") or date) for item in etf_flows),
                source_type="derived",
                status="proxy",
                confidence=0.55,
                note="Volume and price activity proxy; it is not a reported ETF fund-flow series.",
            )
        )
    else:
        provenance.append(unavailable_record("Yahoo Finance ETF activity proxy", date, "No ETF activity rows were available."))

    short_meta = (short_sell or {}).get("meta", {}) or {}
    short_status = str((short_sell or {}).get("status", "unavailable") or "unavailable")
    provenance.append(
        provenance_record(
            source_name=str(short_meta.get("source") or "HKEX short-selling data"),
            source_url=str(short_meta.get("source_url") or "https://www.hkex.com.hk/"),
            as_of=str(short_meta.get("effective_date") or date),
            source_type="official",
            status=short_status,
            confidence=0.95 if short_status == "ok" else 0.0,
            note="Official HKEX short-selling turnover when available.",
        )
    )
    has_short_sell = short_status == "ok"
    status = "ok" if has_short_sell and etf_flows else "partial" if has_short_sell or etf_flows else "unavailable"

    return {
        "status": status,
        "premarket_movers": movers,
        "etf_flows": etf_flows,
        "block_trades": block_trades,
        "unusual_options": options,
        "short_sell": short_sell,
        "formatted_text": analyzer.format_for_report(movers, etf_flows, block_trades, options),
        "provenance": provenance,
    }
