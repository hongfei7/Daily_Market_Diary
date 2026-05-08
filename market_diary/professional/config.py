from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional


DEFAULT_PROFESSIONAL_CONFIG: Dict[str, Any] = {
    "system": {
        "name": "Morning Research Workbench",
        "version": "3.0.0",
        "timezone": "Asia/Shanghai",
        "output_dir": "reports_professional",
    },
    "calendar": {
        "primary_market": "HKEX",
        "closed_weekdays": [5, 6],
        "closed_dates": [],
    },
    "report": {
        "language": "en",
        "quick_items_limit": 10,
        "quick_macro_events": 5,
        "quick_watchlist_items_per_bucket": 1,
        "catalyst_window_days": 7,
        "watchlist_news_limit": 2,
        "watchlist_story_limit": 2,
        "watchlist_workers": 4,
        "top_macro_events": 6,
        "top_news_items": 8,
        "top_movers": 6,
        "top_high_frequency_items": 8,
        "top_catalysts": 10,
        "top_source_links": 15,
        "show_internal_reflection": False,
    },
    "llm": {
        "enabled": True,
        "max_retries": 2,
        "max_workers": 4,
        "retry_base_delay_seconds": 2.0,
        "retry_backoff_multiplier": 2.0,
        "retry_max_delay_seconds": 18.0,
        "provider_parallelism": {
            "minimax": 1,
        },
        "routes": {
            "default_model": {
                "env": "LLM_MODEL",
                "fallback": "MiniMax-M2.7",
            },
            "fast_model": {
                "env": "LLM_FAST_MODEL",
                "fallback": "MiniMax-M2.7",
            },
            "deep_model": {
                "env": "LLM_DEEP_MODEL",
                "fallback": "MiniMax-M2.7",
            },
            "future_deep_model": {
                "env": "LLM_FUTURE_DEEP_MODEL",
                "fallback": "Opus-4.6",
            },
        },
        "tasks": {
            "news_selection": {
                "enabled": True,
                "route": "default_model",
                "temperature": 0.0,
                "max_tokens": 1400,
            },
            "overnight_review": {
                "enabled": True,
                "route": "default_model",
                "temperature": 0.1,
                "max_tokens": 1400,
            },
            "hk_review": {
                "enabled": True,
                "route": "default_model",
                "temperature": 0.1,
                "max_tokens": 1400,
            },
            "macro_interpretation": {
                "enabled": True,
                "route": "default_model",
                "temperature": 0.0,
                "max_tokens": 1100,
            },
            "company_commentary": {
                "enabled": True,
                "route": "default_model",
                "temperature": 0.0,
                "max_tokens": 1400,
            },
            "theme_deep_dive": {
                "enabled": True,
                "route": "default_model",
                "temperature": 0.1,
                "max_tokens": 1600,
            },
            "final_framing": {
                "enabled": True,
                "route": "default_model",
                "temperature": 0.0,
                "max_tokens": 1200,
            },
        },
    },
    "macro_indicator_map": {
        "CPI": {
            "impact": "Rates expectations, the dollar, and growth-equity valuation",
            "industries": ["Technology", "Internet", "Brokers"],
            "beat_direction": "Hot inflation would pressure duration assets",
            "miss_direction": "Cooling inflation would support lower-rate trades",
        },
        "PPI": {
            "impact": "Inflation transmission, upstream resources, and manufacturing costs",
            "industries": ["Metals", "Chemicals", "Manufacturing"],
            "beat_direction": "Higher input costs would support resource-linked assets",
            "miss_direction": "Softer input pressure would help downstream margins",
        },
        "PMI": {
            "impact": "Growth direction, cyclicals, and manufacturing-chain pricing",
            "industries": ["Machinery", "Building materials", "Industrials"],
            "beat_direction": "A stronger PMI would favor cyclicals",
            "miss_direction": "A softer PMI would tilt leadership toward defensives",
        },
        "Nonfarm": {
            "impact": "Treasury yields, the dollar, and overall risk appetite",
            "industries": ["Technology", "Financials", "Consumer discretionary"],
            "beat_direction": "A strong jobs print would lift the rates floor",
            "miss_direction": "A weak jobs print would revive easing expectations",
        },
        "Retail Sales": {
            "impact": "Consumer resilience, growth expectations, and risk-asset pricing",
            "industries": ["Consumer", "E-commerce", "Consumer discretionary"],
            "beat_direction": "A strong print would support the consumer complex",
            "miss_direction": "A weak print would raise pressure on end-demand assumptions",
        },
        "FOMC": {
            "impact": "Global liquidity, the rates curve, and growth style",
            "industries": ["Technology", "Financials", "Gold"],
            "beat_direction": "A hawkish outcome would support the dollar and front-end yields",
            "miss_direction": "A dovish outcome would help growth and precious metals",
        },
        "Aggregate Financing": {
            "impact": "Credit expansion, domestic demand chains, and financials",
            "industries": ["Banks", "Property chain", "Infrastructure"],
            "beat_direction": "A stronger credit pulse would help domestic-demand cyclicals",
            "miss_direction": "Soft credit would keep stimulus expectations under review",
        },
        "Exports": {
            "impact": "External demand, FX, and manufacturing orders",
            "industries": ["Export manufacturing", "Shipping", "Electronics"],
            "beat_direction": "A resilient export print would support global-demand proxies",
            "miss_direction": "A weak export print would pressure export-linked names",
        },
    },
    "watchlists": {
        "core_coverage": [
            {
                "ticker": "0700.HK",
                "name": "Tencent",
                "sector": "Internet",
                "thesis": "Core offshore China platform proxy for gaming, ads, and AI monetization",
                "upcoming_catalyst": "Game grossing trends and ad-demand checks",
            },
            {
                "ticker": "9988.HK",
                "name": "Alibaba",
                "sector": "Internet",
                "thesis": "Key read-through for China consumption, cloud, and platform regulation",
                "upcoming_catalyst": "Cloud demand and GMV updates",
            },
            {
                "ticker": "3690.HK",
                "name": "Meituan",
                "sector": "Internet",
                "thesis": "High-frequency read on local services demand and platform competition",
                "upcoming_catalyst": "Order trends and margin commentary",
            },
            {
                "ticker": "0388.HK",
                "name": "HKEX",
                "sector": "Exchange",
                "thesis": "Direct proxy for Hong Kong market turnover, listings, and southbound interest",
                "upcoming_catalyst": "Turnover data and IPO pipeline updates",
            },
        ],
        "focus_pool": [
            {
                "ticker": "1211.HK",
                "name": "BYD Company",
                "sector": "Autos",
                "thesis": "Hong Kong-listed proxy for EV volumes, export mix, and price competition",
                "upcoming_catalyst": "Weekly sales data and model rollout cadence",
            },
            {
                "ticker": "1299.HK",
                "name": "AIA",
                "sector": "Insurance",
                "thesis": "Read-through for wealth demand, mainland visitor activity, and HK financial sentiment",
                "upcoming_catalyst": "NBV trends and agency momentum",
            },
            {
                "ticker": "0941.HK",
                "name": "China Mobile",
                "sector": "Telecom",
                "thesis": "Defensive SOE benchmark for yield trade and domestic digital-infrastructure spend",
                "upcoming_catalyst": "Capex updates and dividend policy signals",
            },
        ],
        "learning_pool": [
            {
                "ticker": "2800.HK",
                "name": "Tracker Fund of Hong Kong",
                "sector": "Hong Kong beta",
                "thesis": "Clean listed proxy for broad HSI positioning and passive flows",
                "upcoming_catalyst": "Index-turnover shifts and ETF flows",
            },
            {
                "ticker": "2828.HK",
                "name": "HSCEI ETF",
                "sector": "H-shares",
                "thesis": "Useful proxy for state-owned and old-economy H-share leadership",
                "upcoming_catalyst": "Policy flow-through and financial/energy leadership",
            },
            {
                "ticker": "3033.HK",
                "name": "Hang Seng TECH ETF",
                "sector": "HK growth",
                "thesis": "Fast proxy for Hong Kong internet and growth leadership",
                "upcoming_catalyst": "AI narrative shifts and China platform regulation headlines",
            },
        ],
    },
    "thinking": {
        "rotation": [
            {
                "weekday": 0,
                "theme": "AI and Compute Chain",
                "angle": "Track whether global AI capex, semis, cloud, and server demand are still carrying offshore China internet and hardware sentiment.",
                "keywords": ["ai", "gpu", "semiconductor", "server", "cloud", "data center", "compute"],
            },
            {
                "weekday": 1,
                "theme": "China Consumption Recovery",
                "angle": "Focus on whether travel, retail, internet local services, and premium consumption data still support a demand-repair narrative.",
                "keywords": ["consumer", "retail", "travel", "luxury", "macau", "local services", "food delivery"],
            },
            {
                "weekday": 2,
                "theme": "Innovative Biotech and Out-Licensing",
                "angle": "Watch whether clinical data, approvals, and licensing momentum are still supporting the innovative-healthcare rerating story.",
                "keywords": ["biotech", "pharma", "drug", "clinical", "approval", "licensing", "medical"],
            },
            {
                "weekday": 3,
                "theme": "China Exports and Supply-Chain Relocation",
                "angle": "Track tariffs, trade policy, shipping, and manufacturing orders to see whether export resilience is still the cleaner China macro expression.",
                "keywords": ["export", "shipping", "logistics", "manufacturing", "tariff", "trade", "supply chain"],
            },
            {
                "weekday": 4,
                "theme": "Hong Kong Market Structure and Flows",
                "angle": "Focus on turnover, Stock Connect, ETF flows, HKEX activity, and style leadership to understand whether Hong Kong is being driven by fundamentals or positioning.",
                "keywords": ["hong kong", "hkex", "flow", "southbound", "northbound", "turnover", "etf"],
            },
        ],
        "reflection_prompts": [
            "What was the most important overnight surprise, and does it change my base case?",
            "Which sector or stock view now deserves a tighter follow-up or a revised assumption?",
            "If I were asked for a two-minute market view this morning, what would my answer be?",
        ],
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return payload


def load_professional_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_PROFESSIONAL_CONFIG)

    candidates = []
    if config_path:
        candidates.append(config_path)
    else:
        if os.path.exists("config.json"):
            candidates.append("config.json")

    for candidate in candidates:
        if os.path.exists(candidate):
            config = _deep_merge(config, _load_json_file(candidate))
            config["config_path"] = os.path.abspath(candidate)
            break

    output_dir = os.getenv("OUTPUT_DIR")
    if output_dir:
        config.setdefault("system", {})["output_dir"] = output_dir

    timezone = os.getenv("TIMEZONE")
    if timezone:
        config.setdefault("system", {})["timezone"] = timezone

    return config
