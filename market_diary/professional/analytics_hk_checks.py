from __future__ import annotations

from typing import Any, Dict, List

from market_diary.professional.analytics_market import _format_signed, _parse_float, _parse_pct, _summary_item


def _format_price_or_na(value: Any, digits: int = 2) -> str:
    price = _parse_float(value)
    if price is None:
        return "N/A"
    return f"{price:.{digits}f}"


def _summarize_hk_etf_proxy(movers_data: Dict[str, Any]) -> str:
    hk_flows = []
    for item in (movers_data or {}).get("etf_flows", []) or []:
        ticker = item.get("ticker", "")
        if ticker in {"2800.HK", "2828.HK", "3033.HK"}:
            hk_flows.append(
                f"{ticker} {item.get('change_pct', 0):+.2f}% on {item.get('volume_ratio', 1):.2f}x volume ({item.get('estimated_flow_direction', 'n/a')})"
            )
    if not hk_flows:
        return "Hong Kong ETF proxy detail was not refreshed for this run."
    return "; ".join(hk_flows[:3])


def _usdhkd_band_read(price: Any) -> str:
    level = _parse_float(price)
    if level is None:
        return "USD/HKD spot was not refreshed in the current quote set."
    if level >= 7.845:
        return "Close to the weak-side Convertibility Undertaking; keep HKMA liquidity operations in focus."
    if level <= 7.755:
        return "Close to the strong-side Convertibility Undertaking; watch for liquidity absorption or funding shifts."
    return "Inside the linked-exchange band without immediate boundary stress."


def _adapter_metric(adapter_data: Dict[str, Any], key: str) -> Dict[str, Any]:
    item = (adapter_data or {}).get(key, {})
    return item if isinstance(item, dict) else {}


def _quick_check_row(
    metric: str,
    value: str,
    status: str,
    note: str,
    source: str = "",
    as_of: str = "",
) -> Dict[str, str]:
    return {
        "metric": metric,
        "value": value,
        "status": status,
        "note": note,
        "source": source,
        "as_of": as_of,
    }


def build_hk_quick_checks(
    summary: Dict[str, Any],
    movers_data: Dict[str, Any],
    hk_desk_view: Dict[str, Any],
    hk_local_data: Dict[str, Any],
) -> List[Dict[str, str]]:
    usdhkd_item = _summary_item(summary, "FX", "USD/HKD")
    hsi_item = _summary_item(summary, "Equities", "Hang Seng Index")
    hstech_item = _summary_item(summary, "Equities", "Hang Seng TECH ETF")
    band_metric = _adapter_metric(hk_local_data, "linked_exchange_band")
    turnover_metric = _adapter_metric(hk_local_data, "main_board_turnover")
    turnover_ratio_metric = _adapter_metric(hk_local_data, "turnover_vs_20d")
    hibor_metric = _adapter_metric(hk_local_data, "hibor_1m")
    aggregate_metric = _adapter_metric(hk_local_data, "aggregate_balance")
    base_rate_metric = _adapter_metric(hk_local_data, "base_rate")
    short_selling_metric = _adapter_metric(hk_local_data, "short_selling_ratio")
    southbound_metric = _adapter_metric(hk_local_data, "southbound_net_flow")
    northbound_metric = _adapter_metric(hk_local_data, "northbound_net_flow")
    ah_metric = _adapter_metric(hk_local_data, "ah_premium_index")

    usdhkd_value = _format_price_or_na(usdhkd_item.get("Price"), digits=4)
    band_value = band_metric.get("display_value", "")
    band_note = band_metric.get("note", "")
    usdhkd_note = _usdhkd_band_read(usdhkd_item.get("Price"))
    if band_value and band_value != "N/A":
        usdhkd_display = f"{usdhkd_value} | band {band_value}"
        usdhkd_status = "live_hybrid"
        usdhkd_source = "Yahoo Finance + HKMA linked band"
        usdhkd_note = f"{usdhkd_note} {band_note}".strip()
        usdhkd_as_of = usdhkd_item.get("As Of", "") or band_metric.get("as_of", "")
    else:
        usdhkd_display = usdhkd_value
        usdhkd_status = "live_quote" if usdhkd_value != "N/A" else "unavailable"
        usdhkd_source = "Yahoo Finance"
        usdhkd_as_of = usdhkd_item.get("As Of", "")

    turnover_display = turnover_ratio_metric.get("display_value") or turnover_metric.get("display_value", "N/A")
    turnover_note = turnover_ratio_metric.get("note") or turnover_metric.get("note") or "Turnover context was not refreshed for this run."
    flow_value = "N/A"
    flow_note = southbound_metric.get("note", "") or northbound_metric.get("note", "")
    flow_status = "unavailable"
    flow_source = ""
    flow_as_of = ""
    southbound_status = southbound_metric.get("status", "unavailable")
    northbound_status = northbound_metric.get("status", "unavailable")
    if southbound_status != "unavailable" or northbound_status != "unavailable":
        flow_value = f"Southbound {southbound_metric.get('display_value', 'N/A')} | Northbound {northbound_metric.get('display_value', 'N/A')}"
        flow_status = southbound_status if southbound_status != "unavailable" else northbound_status
        flow_source = southbound_metric.get("source", "") or northbound_metric.get("source", "")
        flow_as_of = southbound_metric.get("as_of", "") or northbound_metric.get("as_of", "")

    rows = [
        _quick_check_row(
            metric="Main Board turnover vs 20D",
            value=turnover_display,
            status=turnover_metric.get("status", turnover_ratio_metric.get("status", "unavailable")),
            note=turnover_note,
            source=turnover_metric.get("source", ""),
            as_of=turnover_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="HIBOR 1M",
            value=hibor_metric.get("display_value", "N/A"),
            status=hibor_metric.get("status", "unavailable"),
            note=hibor_metric.get("note", "Hong Kong funding data was not refreshed for this run."),
            source=hibor_metric.get("source", ""),
            as_of=hibor_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="Aggregate Balance",
            value=aggregate_metric.get("display_value", "N/A"),
            status=aggregate_metric.get("status", "unavailable"),
            note=aggregate_metric.get("note", "Aggregate Balance data was not refreshed for this run."),
            source=aggregate_metric.get("source", ""),
            as_of=aggregate_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="Base Rate / linked band",
            value=f"Base rate {base_rate_metric.get('display_value', 'N/A')} | band {band_metric.get('display_value', 'N/A')}",
            status=base_rate_metric.get("status", band_metric.get("status", "unavailable")),
            note=base_rate_metric.get("note", "") or band_metric.get("note", "") or "Linked-rate policy context was not refreshed for this run.",
            source=base_rate_metric.get("source", "") or band_metric.get("source", ""),
            as_of=base_rate_metric.get("as_of", "") or band_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="USD/HKD spot vs band",
            value=usdhkd_display,
            status=usdhkd_status,
            note=usdhkd_note,
            source=usdhkd_source,
            as_of=usdhkd_as_of,
        ),
        _quick_check_row(
            metric="Short-selling ratio",
            value=short_selling_metric.get("display_value", "N/A"),
            status=short_selling_metric.get("status", "unavailable"),
            note=short_selling_metric.get("note", "Short-selling ratio was not refreshed for this run."),
            source=short_selling_metric.get("source", ""),
            as_of=short_selling_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="Southbound / Northbound net flow",
            value=flow_value,
            status=flow_status,
            note=flow_note or "Stock Connect public data were incomplete for this report date.",
            source=flow_source,
            as_of=flow_as_of,
        ),
        _quick_check_row(
            metric="AH premium index",
            value=ah_metric.get("display_value", "N/A"),
            status=ah_metric.get("status", "unavailable"),
            note=ah_metric.get("note", "A/H premium could not be calculated from the configured public quote set."),
            source=ah_metric.get("source", ""),
            as_of=ah_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="Hong Kong leadership",
            value=hk_desk_view.get("headline") or hk_desk_view.get("leadership", "N/A"),
            status="proxy",
            note="Use HSI / HSCEI / 3033.HK ETF relative moves as the opening style read.",
            source="HSI / HSCEI / 3033.HK ETF relative performance",
            as_of=usdhkd_item.get("As Of", ""),
        ),
        _quick_check_row(
            metric="HSI vs 3033.HK ETF",
            value=f"HSI {_format_signed(_parse_pct(hsi_item.get('Pct Change')))} | 3033.HK ETF {_format_signed(_parse_pct(hstech_item.get('Pct Change')))}",
            status="proxy",
            note="This is the fastest check for whether the day is beta-led or growth-led in Hong Kong.",
            source="Yahoo Finance summary snapshot",
            as_of=hsi_item.get("As Of", "") or hstech_item.get("As Of", ""),
        ),
        _quick_check_row(
            metric="HK ETF flow proxy",
            value=_summarize_hk_etf_proxy(movers_data),
            status="proxy",
            note="Use only as a secondary lens when official Stock Connect flow evidence is incomplete.",
            source="Hong Kong ETF volume proxy",
            as_of="",
        ),
    ]
    return rows
