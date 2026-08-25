from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from market_diary.modules.llm_client import (
    api_key_available,
    get_available_providers,
    get_client,
    get_completion_extra_body,
    get_default_model,
    get_default_provider,
)


LLM_SYSTEM_PROMPT = """You are a senior analyst at a Hong Kong Chinese securities research institute.

Write for institutional investors and internal research colleagues.

Rules:
1. Use only the supplied facts. Do not invent data, prices, companies, events, or causal claims.
2. Separate facts from interpretation. When making a judgement, use bounded analytical language.
3. Return strict JSON only, with exactly the keys requested.
4. Be concise, specific, and professional. Avoid slogans and generic market commentary.
5. If the provided facts are insufficient, say so plainly instead of guessing.
"""


TASK_ORDER = [
    "news_selection",
    "overnight_review",
    "hk_review",
    "macro_interpretation",
    "company_commentary",
    "theme_deep_dive",
    "final_framing",
]

TASK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "news_selection": {
        "summary": "",
        "selected_news": [],
    },
    "overnight_review": {
        "paragraph": "",
        "drivers": [],
        "hk_open_implication": "",
    },
    "hk_review": {
        "paragraph": "",
        "local_leadership": "",
        "follow_through": "",
    },
    "macro_interpretation": {
        "paragraph": "",
        "watchpoints": [],
    },
    "company_commentary": {
        "paragraph": "",
        "company_notes": [],
    },
    "theme_deep_dive": {
        "paragraph": "",
        "watch_items": [],
    },
    "final_framing": {
        "one_line_market_pulse": "",
        "thinking_note": "",
        "risk_check": "",
        "interview_answer": "",
    },
}

TASK_FEW_SHOTS: Dict[str, str] = {
    "news_selection": """Example JSON:
{"summary":"Overnight news flow tilted constructive for Hong Kong because softer US rates and internet-related headlines mattered more than isolated defensive news.","selected_news":[{"headline":"Example headline","why_it_matters":"It changes the market's earnings or policy framing.","hk_market_impact":"Most relevant for Hong Kong internet and growth sentiment.","importance":"A"}]}""",
    "overnight_review": """Example JSON:
{"paragraph":"US equities rose because softer inflation and lower yields supported duration-sensitive sectors, while oil stabilized enough to keep the geopolitical premium contained. For Hong Kong, the key question is whether offshore China proxies confirm that the move was broad risk appetite rather than only US mega-cap leadership.","drivers":["Softer inflation reduced rates pressure.","Lower yields supported growth and internet names."],"hk_open_implication":"A constructive open is more credible if the 3033.HK Hang Seng TECH ETF proxy and FXI also confirm the move."}""",
    "hk_review": """Example JSON:
{"paragraph":"Hong Kong and A-share follow-through should be judged through style leadership, not index direction alone. If the 3033.HK Hang Seng TECH ETF proxy outperforms HSCEI while USD/CNH stays stable, the market is more likely to read the overnight tape as supportive for growth rather than just broad beta.","local_leadership":"Growth-led if the 3033.HK ETF proxy outperforms HSCEI.","follow_through":"Watch whether CSI 300 and offshore China proxies confirm the same style read."}""",
    "macro_interpretation": """Example JSON:
{"paragraph":"The macro calendar matters because it can quickly reprice the rates-and-dollar backdrop that is supporting today's opening setup. If the data surprise is material, growth leadership could either extend or reverse early in the session.","watchpoints":["US yields and DXY after the release.","Whether Hong Kong growth proxies hold their opening tone."]}""",
    "company_commentary": """Example JSON:
{"paragraph":"Only portfolio-relevant or estimate-changing company events deserve space in the morning decision brief; market-wide low-signal filings should remain aggregated.","company_notes":[{"ticker":"0001.HK","commentary":"Fact: the primary filing reset the profit range. Investor read: test whether the driver is recurring and how far it sits from expectations. Next check: update the earnings bridge before changing the thesis."}]}""",
    "theme_deep_dive": """Example JSON:
{"paragraph":"The weekly theme still deserves attention because recent data points are starting to line up rather than remaining isolated headlines. The right question is whether the current signals are strong enough to move positioning, or whether they are only useful as watchlist preparation for the next catalyst window.","watch_items":["Signal one to verify.","Catalyst two to monitor."]}""",
    "final_framing": """Example JSON:
{"one_line_market_pulse":"Softer dollar pressure and steadier offshore China proxies left the overnight setup mildly constructive for Hong Kong, but local flow confirmation still matters.","thinking_note":"Treat today's opening setup as conditional rather than fully confirmed: if Hong Kong growth leadership broadens with a stable USD/CNH backdrop, the overnight signal is becoming investable rather than merely interesting.","risk_check":"A fast reversal in yields, the dollar, or offshore China proxies would weaken the constructive opening narrative.","interview_answer":"The setup is constructive but not one-way. I would frame today as a confirmation test for Hong Kong growth leadership rather than a blind risk-on call."}""",
}

TaskRunner = Callable[[str, Dict[str, Any], str], Tuple[Dict[str, Any], Dict[str, Any]]]
LLM_CACHE_VERSION = "v2"

# A truncated response means the budget ran out, so the retry needs more of
# it. The ceiling keeps a runaway prompt from becoming an expensive loop.
TRUNCATION_RETRY_MULTIPLIER = 1.6
MAX_TOKENS_CEILING = 8000


def _api_key_present() -> bool:
    return api_key_available()


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    try:
        payload = ast.literal_eval(text)
        return payload if isinstance(payload, dict) else {}
    except (ValueError, SyntaxError):
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            try:
                payload = ast.literal_eval(candidate)
                return payload if isinstance(payload, dict) else {}
            except (ValueError, SyntaxError):
                return {}
    return {}


def _coerce_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _coerce_string_list(value: Any, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            output.append(item.strip())
        if len(output) >= limit:
            break
    return output


def _coerce_dict_list(value: Any, keys: List[str], limit: int) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    output: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        row = {key: _coerce_string(item.get(key, "")) for key in keys}
        if any(row.values()):
            output.append(row)
        if len(output) >= limit:
            break
    return output


def _coerce_task_payload(task_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    defaults = TASK_DEFAULTS.get(task_name, {})
    if task_name == "news_selection":
        return {
            "summary": _coerce_string(payload.get("summary")),
            "selected_news": _coerce_dict_list(
                payload.get("selected_news"),
                ["headline", "why_it_matters", "hk_market_impact", "importance"],
                limit=5,
            ),
        }
    if task_name == "overnight_review":
        return {
            "paragraph": _coerce_string(payload.get("paragraph")),
            "drivers": _coerce_string_list(payload.get("drivers"), limit=4),
            "hk_open_implication": _coerce_string(payload.get("hk_open_implication")),
        }
    if task_name == "hk_review":
        return {
            "paragraph": _coerce_string(payload.get("paragraph")),
            "local_leadership": _coerce_string(payload.get("local_leadership")),
            "follow_through": _coerce_string(payload.get("follow_through")),
        }
    if task_name == "macro_interpretation":
        return {
            "paragraph": _coerce_string(payload.get("paragraph")),
            "watchpoints": _coerce_string_list(payload.get("watchpoints"), limit=4),
        }
    if task_name == "company_commentary":
        return {
            "paragraph": _coerce_string(payload.get("paragraph")),
            "company_notes": _coerce_dict_list(payload.get("company_notes"), ["ticker", "commentary"], limit=6),
        }
    if task_name == "theme_deep_dive":
        return {
            "paragraph": _coerce_string(payload.get("paragraph")),
            "watch_items": _coerce_string_list(payload.get("watch_items"), limit=5),
        }
    if task_name == "final_framing":
        return {
            "one_line_market_pulse": _coerce_string(payload.get("one_line_market_pulse")),
            "thinking_note": _coerce_string(payload.get("thinking_note")),
            "risk_check": _coerce_string(payload.get("risk_check")),
            "interview_answer": _coerce_string(payload.get("interview_answer")),
        }
    return defaults.copy()


def _payload_has_content(payload: Dict[str, Any]) -> bool:
    for value in (payload or {}).values():
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list) and value:
            return True
        if isinstance(value, dict) and _payload_has_content(value):
            return True
    return False


def _route_config(llm_config: Dict[str, Any], route_name: str) -> Dict[str, Any]:
    return ((llm_config.get("routes", {}) or {}).get(route_name, {}) or {})


def _route_fallback_model(route: Dict[str, Any], provider: str = "") -> str:
    fallback = route.get("fallback")
    selected_provider = provider or get_default_provider()
    if isinstance(fallback, dict):
        model = (
            fallback.get(selected_provider)
            or fallback.get("default")
            or get_default_model(selected_provider)
        )
        return str(model).strip() or get_default_model(selected_provider)
    if fallback:
        return str(fallback).strip() or get_default_model(selected_provider)
    return get_default_model(selected_provider)


def _resolve_model(llm_config: Dict[str, Any], task_name: str) -> Tuple[str, str]:
    task_config = ((llm_config.get("tasks", {}) or {}).get(task_name, {}) or {})
    route_name = str(task_config.get("route", "default_model"))
    route = _route_config(llm_config, route_name)
    env_name = route.get("env", "") or ("LLM_MODEL" if route_name == "default_model" else "")
    env_value = os.getenv(env_name, "").strip() if env_name else ""
    model = env_value or _route_fallback_model(route)
    return model, route_name


def _model_candidates(llm_config: Dict[str, Any], task_name: str) -> List[Tuple[str, str, str]]:
    task_config = _task_config(llm_config, task_name)
    preferred_provider = str(task_config.get("provider", "") or "").strip().lower()
    model, route_name = _resolve_model(llm_config, task_name)
    primary_provider = get_default_provider()
    available_providers = get_available_providers()
    if preferred_provider and preferred_provider in available_providers:
        route = _route_config(llm_config, route_name)
        candidates = [
            (preferred_provider, _route_fallback_model(route, preferred_provider), f"{route_name}:preferred")
        ]
        ordered_fallbacks = [primary_provider] + [provider for provider in available_providers if provider != primary_provider]
        for fallback_provider in ordered_fallbacks:
            if fallback_provider == preferred_provider or fallback_provider not in available_providers:
                continue
            fallback_model = _route_fallback_model(route, fallback_provider)
            candidates.append((fallback_provider, fallback_model, f"{route_name}:fallback"))
        return candidates

    candidates = [(primary_provider, model, route_name)]
    for fallback_provider in available_providers:
        if fallback_provider == primary_provider:
            continue
        route = _route_config(llm_config, route_name)
        fallback_model = _route_fallback_model(route, fallback_provider)
        candidates.append((fallback_provider, fallback_model, f"{route_name}:fallback"))
    return candidates


def _provider_cap_for_model(provider_caps: Dict[str, Any], model_name: str) -> Optional[int]:
    normalized = (model_name or "").strip().lower()
    if "minimax" in normalized:
        cap = provider_caps.get("minimax")
        if cap is not None:
            try:
                return max(int(cap), 1)
            except (TypeError, ValueError):
                return None
    return None


def _task_config(llm_config: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    return ((llm_config.get("tasks", {}) or {}).get(task_name, {}) or {})


def _task_enabled(llm_config: Dict[str, Any], task_name: str) -> bool:
    return bool(_task_config(llm_config, task_name).get("enabled", True))


def _extract_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                    continue
                if item.get("type") == "text":
                    nested = item.get("content")
                    if isinstance(nested, str) and nested.strip():
                        parts.append(nested.strip())
        return "\n".join(parts).strip()
    return ""


def _choice_finish_reason(response: Any) -> str:
    try:
        choices = getattr(response, "choices", []) or []
        if not choices:
            return ""
        return str(getattr(choices[0], "finish_reason", "") or "").strip().lower()
    except Exception:
        return ""


def _llm_response_looks_truncated(raw: str, finish_reason: str = "") -> bool:
    """Detect provider-side truncation before partial JSON can be cached."""
    text = str(raw or "").strip()
    reason = str(finish_reason or "").strip().lower()
    if reason in {"length", "max_tokens"}:
        return True
    if not text:
        return False
    if text.endswith(("...", "…")):
        return True
    if text.count("{") > text.count("}"):
        return True
    return False


# Ordered most specific first: the first matching class wins.
_ERROR_CLASS_MARKERS: List[tuple] = [
    ("auth", ("401", "403", "unauthorized", "forbidden", "invalid api key", "authentication")),
    ("rate_limit", ("429", "rate limit", "too many requests", "quota")),
    ("overloaded", ("529", "overloaded", "temporarily unavailable")),
    ("timeout", ("timeout", "timed out", "deadline exceeded")),
    ("connection", ("connection reset", "connection aborted", "connection error", "ssl", "dns")),
    ("http_5xx", ("500", "502", "503", "504", "internal server error", "bad gateway")),
    ("truncated", ("truncated", "max_tokens", "finish_reason=length")),
    ("json_parse", ("jsondecodeerror", "expecting value", "invalid json", "could not parse")),
    ("schema_mismatch", ("keyerror", "missing field", "schema", "validationerror")),
    ("bad_request", ("400", "invalid request", "unsupported")),
]


def _token_usage(response: Any, max_tokens: int) -> Dict[str, Any]:
    """Capture token usage so budget questions can be measured, not guessed.

    Nothing recorded usage before, so a run of truncation failures gave no way
    to tell whether the budget was genuinely too small or the request was being
    routed to a model that spends the budget on reasoning.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"max_tokens": max_tokens}

    def _get(name: str) -> Optional[int]:
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    completion = _get("completion_tokens")
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = None
    if details is not None:
        reasoning = getattr(details, "reasoning_tokens", None)
        if reasoning is None and isinstance(details, dict):
            reasoning = details.get("reasoning_tokens")

    payload: Dict[str, Any] = {
        "prompt_tokens": _get("prompt_tokens"),
        "completion_tokens": completion,
        "total_tokens": _get("total_tokens"),
        "max_tokens": max_tokens,
    }
    if reasoning is not None:
        payload["reasoning_tokens"] = reasoning
    if completion is not None and max_tokens:
        payload["budget_used_pct"] = round(completion / max_tokens * 100.0, 1)
    return payload


def classify_error(message: str) -> str:
    """Bucket a failure so recurring causes can be counted across runs.

    The run summary previously reported only "4 error(s)", which cannot
    distinguish a provider outage from a prompt that never parses, so there was
    no way to tell which fix would help.
    """
    lowered = (message or "").lower()
    if not lowered.strip():
        return "unknown"
    for label, markers in _ERROR_CLASS_MARKERS:
        if any(marker in lowered for marker in markers):
            return label
    return "other"


def _is_retryable_error(message: str) -> bool:
    lowered = (message or "").lower()
    retry_markers = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "529",
        "rate limit",
        "overloaded",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
    ]
    return any(marker in lowered for marker in retry_markers)


def _retry_delay_seconds(llm_config: Dict[str, Any], attempt: int, error_message: str) -> float:
    base = float(llm_config.get("retry_base_delay_seconds", 2.0))
    multiplier = float(llm_config.get("retry_backoff_multiplier", 2.0))
    max_delay = float(llm_config.get("retry_max_delay_seconds", 20.0))
    if "529" in (error_message or "") or "overloaded" in (error_message or "").lower():
        base = max(base, 3.0)
    delay = base * (multiplier ** max(attempt - 1, 0))
    return min(delay, max_delay)


def _effective_max_workers(llm_config: Dict[str, Any], task_names: Optional[List[str]] = None) -> int:
    requested = max(int(llm_config.get("max_workers", 4)), 1)
    provider_caps = (llm_config.get("provider_parallelism", {}) or {})
    selected_tasks = task_names or list((llm_config.get("tasks", {}) or {}).keys()) or ["default_model"]

    caps = []
    has_uncapped_task = False
    for task_name in selected_tasks:
        resolved_task = task_name if task_name in (llm_config.get("tasks", {}) or {}) else "default_model"
        candidates = _model_candidates(llm_config, resolved_task)
        model_name = candidates[0][1] if candidates else _resolve_model(llm_config, resolved_task)[0]
        cap = _provider_cap_for_model(provider_caps, model_name)
        if cap is not None:
            caps.append(cap)
        else:
            has_uncapped_task = True
    return requested if has_uncapped_task else min([requested, *caps]) if caps else requested


def _hash_context(task_name: str, context: Dict[str, Any], model: str, prompt: str) -> str:
    payload = json.dumps(
        {
            "version": LLM_CACHE_VERSION,
            "task": task_name,
            "model": model,
            "prompt": prompt,
            "system_prompt": LLM_SYSTEM_PROMPT,
            "context": context,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str, task_name: str, context: Dict[str, Any], model: str, prompt: str) -> str:
    digest = _hash_context(task_name, context, model, prompt)
    return os.path.join(cache_dir, f"{task_name}_{digest}.json")


def _load_cache(cache_dir: str, task_name: str, context: Dict[str, Any], model: str, prompt: str) -> Optional[Dict[str, Any]]:
    if not cache_dir:
        return None
    path = _cache_path(cache_dir, task_name, context, model, prompt)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _save_cache(cache_dir: str, task_name: str, context: Dict[str, Any], model: str, prompt: str, payload: Dict[str, Any]) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, task_name, context, model, prompt)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _market_snapshot(bundle: Dict[str, Any], limit: int = 12) -> List[Dict[str, Any]]:
    return ((bundle.get("overview", {}) or {}).get("snapshot_rows", []) or [])[:limit]


def _news_candidates(bundle: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    return ((bundle.get("sector_digest", {}) or {}).get("graded_news", []) or [])[:limit]


def _macro_items(bundle: Dict[str, Any], limit: int = 6) -> List[Dict[str, Any]]:
    return (bundle.get("macro_agenda", []) or [])[:limit]


def _theme_context(bundle: Dict[str, Any]) -> Dict[str, Any]:
    theme = (bundle.get("theme_deep_dive", {}) or {})
    return {
        "theme": theme.get("theme", ""),
        "angle": theme.get("angle", ""),
        "signals": (theme.get("signals", []) or [])[:4],
        "related_names": (theme.get("related_names", []) or [])[:4],
        "upcoming": (theme.get("upcoming", []) or [])[:4],
        "news": (theme.get("news", []) or [])[:3],
    }


def _weekly_review_context(bundle: Dict[str, Any]) -> Dict[str, Any]:
    weekly = bundle.get("weekly_review", {}) or {}
    if not weekly:
        return {}
    trend_summary = weekly.get("trend_summary", {}) or {}
    return {
        "window": weekly.get("window", {}) or {},
        "summary": weekly.get("summary", ""),
        "trend_summary": {
            "status": trend_summary.get("status", ""),
            "window": trend_summary.get("window", {}) or {},
            "rows": (trend_summary.get("rows", []) or [])[:5],
        },
        "flow_lines": (weekly.get("flow_lines", []) or [])[:4],
        "desk_questions": (weekly.get("desk_questions", []) or [])[:5],
        "key_developments": (weekly.get("developments", []) or [])[:5],
        "next_week": (weekly.get("next_week", []) or [])[:6],
    }


def _non_trading_focus_context(bundle: Dict[str, Any]) -> Dict[str, Any]:
    focus = bundle.get("non_trading_focus", {}) or {}
    if not focus:
        return {}
    return {
        "summary": focus.get("summary", ""),
        "market_regime": focus.get("market_regime", ""),
        "still_moving": (focus.get("still_moving", []) or [])[:5],
        "event_watch": (focus.get("event_watch", []) or [])[:6],
        "action_items": (focus.get("action_items", []) or [])[:5],
        "next_open": (focus.get("next_open", []) or [])[:5],
    }


def _week_ahead_context(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Monday week-ahead lens: calendar, forecast, and watch list for the week."""
    week = bundle.get("week_ahead", {}) or {}
    if not week:
        return {}
    return {
        "week_start": week.get("week_start", ""),
        "week_end": week.get("week_end", ""),
        "summary": week.get("summary", ""),
        "key_events": [
            {"date": it.get("date", ""), "event": it.get("event", ""), "impact": it.get("impact", "")}
            for it in (week.get("week_events", []) or [])[:5]
        ],
        "calendar": (week.get("week_calendar", []) or [])[:5],
        "forecast": week.get("forecast", {}) or {},
        "watch_items": (week.get("watch_items", []) or [])[:5],
    }


def _day_mode_context(bundle: Dict[str, Any]) -> Dict[str, Any]:
    day_mode = bundle.get("day_mode", {}) or {}
    return {
        "mode": day_mode.get("mode", "trading_daily"),
        "label": day_mode.get("label", "Trading Daily"),
        "note": day_mode.get("note", ""),
        "is_trading_day": bool(day_mode.get("is_trading_day", True)),
        "report_horizon": day_mode.get("report_horizon", "daily"),
        "period_start": day_mode.get("period_start", ""),
        "period_end": day_mode.get("period_end", ""),
        "target_hk_session": day_mode.get(
            "target_hk_session",
            day_mode.get("next_hk_trading_day", ""),
        ),
        "next_hk_trading_day": day_mode.get("next_hk_trading_day", ""),
    }


def _should_run_news_selection(bundle: Dict[str, Any]) -> Tuple[bool, str]:
    if _news_candidates(bundle, limit=3) or (bundle.get("raw_news_headlines", []) or []):
        return True, ""
    return False, "No meaningful news candidates were available."


def _should_run_macro(bundle: Dict[str, Any]) -> Tuple[bool, str]:
    if _macro_items(bundle, limit=3):
        return True, ""
    return False, "Macro agenda was empty."


def _should_run_company(bundle: Dict[str, Any]) -> Tuple[bool, str]:
    company_events = bundle.get("company_events", {}) or {}
    if (
        (company_events.get("earnings", []) or [])
        or (company_events.get("ratings", []) or [])
        or (company_events.get("watchlist_announcements", []) or [])
        or _news_candidates(bundle, limit=3)
    ):
        return True, ""
    return False, "No portfolio-relevant company event or sector signal was available."


def _build_task_context(task_name: str, bundle: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    overview = bundle.get("overview", {}) or {}
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}
    base = {
        "report_date": (bundle.get("meta", {}) or {}).get("report_date", ""),
        "day_mode": _day_mode_context(bundle),
        "date_semantics": bundle.get("date_semantics", {}) or {},
        "market_theme": overview.get("theme", ""),
        "risk_regime": overview.get("risk_regime", ""),
    }

    if task_name == "news_selection":
        base.update(
            {
                "graded_news": _news_candidates(bundle, limit=8),
                "raw_news_headlines": (bundle.get("raw_news_headlines", []) or [])[:10],
                "must_watch": (bundle.get("must_watch", []) or [])[:5],
                "week_ahead": _week_ahead_context(bundle),
            }
        )
        return base

    if task_name == "overnight_review":
        base.update(
            {
                "market_snapshot": _market_snapshot(bundle),
                "chart_read": (overview.get("chart_read", {}) or {}),
                "hk_desk_view": hk_desk_view,
                "hk_quick_checks": (bundle.get("hk_quick_checks", []) or [])[:6],
                "attribution_v1": bundle.get("attribution", {}) or {},
                "weekly_review": _weekly_review_context(bundle),
                "non_trading_focus": _non_trading_focus_context(bundle),
                "news_selection": prior.get("news_selection", TASK_DEFAULTS["news_selection"]),
                "high_frequency": (bundle.get("high_frequency", []) or [])[:5],
            }
        )
        return base

    if task_name == "hk_review":
        base.update(
            {
                "market_snapshot": _market_snapshot(bundle),
                "hk_quick_checks": (bundle.get("hk_quick_checks", []) or [])[:6],
                "attribution_v1": bundle.get("attribution", {}) or {},
                "flow_tracker": bundle.get("flow_tracker", {}) or {},
                "weekly_review": _weekly_review_context(bundle),
                "non_trading_focus": _non_trading_focus_context(bundle),
                "overnight_review": prior.get("overnight_review", TASK_DEFAULTS["overnight_review"]),
                "news_selection": prior.get("news_selection", TASK_DEFAULTS["news_selection"]),
                "etf_flows": ((bundle.get("movers_digest", {}) or {}).get("etf_flows", []) or [])[:6],
                "flow_bullets": ((bundle.get("movers_digest", {}) or {}).get("flow_bullets", []) or [])[:4],
                "watchlists": {key: value[:2] for key, value in (bundle.get("watchlists", {}) or {}).items()},
            }
        )
        return base

    if task_name == "macro_interpretation":
        base.update(
            {
                "macro_agenda": _macro_items(bundle, limit=6),
                "high_frequency": (bundle.get("high_frequency", []) or [])[:4],
                "day_mode_note": _day_mode_context(bundle).get("note", ""),
                "weekly_review": _weekly_review_context(bundle),
                "non_trading_focus": _non_trading_focus_context(bundle),
                "week_ahead": _week_ahead_context(bundle),
            }
        )
        return base

    if task_name == "company_commentary":
        company_events = bundle.get("company_events", {}) or {}
        base.update(
            {
                "earnings": (company_events.get("earnings", []) or [])[:6],
                "ratings": (company_events.get("ratings", []) or [])[:6],
                "hkex_announcements": (company_events.get("announcements", []) or [])[:8],
                "news_candidates": _news_candidates(bundle, limit=6),
                "watchlists": {key: value[:2] for key, value in (bundle.get("watchlists", {}) or {}).items()},
            }
        )
        return base

    if task_name == "theme_deep_dive":
        base.update(
            {
                "theme_context": _theme_context(bundle),
                "high_frequency": (bundle.get("high_frequency", []) or [])[:5],
                "macro_agenda": _macro_items(bundle, limit=4),
                "must_watch": (bundle.get("must_watch", []) or [])[:5],
                "weekly_review": _weekly_review_context(bundle),
                "non_trading_focus": _non_trading_focus_context(bundle),
            }
        )
        return base

    if task_name == "final_framing":
        base.update(
            {
                "hk_quick_checks": (bundle.get("hk_quick_checks", []) or [])[:6],
                "news_selection": prior.get("news_selection", TASK_DEFAULTS["news_selection"]),
                "overnight_review": prior.get("overnight_review", TASK_DEFAULTS["overnight_review"]),
                "hk_review": prior.get("hk_review", TASK_DEFAULTS["hk_review"]),
                "macro_interpretation": prior.get("macro_interpretation", TASK_DEFAULTS["macro_interpretation"]),
                "company_commentary": prior.get("company_commentary", TASK_DEFAULTS["company_commentary"]),
                "theme_deep_dive": prior.get("theme_deep_dive", TASK_DEFAULTS["theme_deep_dive"]),
                "attribution_v1": bundle.get("attribution", {}) or {},
                "flow_tracker": bundle.get("flow_tracker", {}) or {},
                "weekly_review": _weekly_review_context(bundle),
                "non_trading_focus": _non_trading_focus_context(bundle),
                "week_ahead": _week_ahead_context(bundle),
                "must_watch": (bundle.get("must_watch", []) or [])[:6],
            }
        )
        return base

    return base


def _build_prompt(task_name: str, context: Dict[str, Any]) -> str:
    day_mode = context.get("day_mode", {}) or {}
    non_trading_instruction = ""
    if not bool(day_mode.get("is_trading_day", True)):
        if day_mode.get("mode") == "weekly_review":
            non_trading_instruction = (
                "Weekly-review rule: synthesize the completed Hong Kong trading week and next-week preparation. "
                "Use weekly_review.trend_summary and desk_questions when supplied. "
                "Do not frame Saturday as a fresh cash-market session or refer to today's Hong Kong open. "
                "Separate weekly evidence, bounded interpretation, and next-week preparation.\n"
            )
        elif day_mode.get("mode") == "week_ahead":
            non_trading_instruction = (
                "Week-ahead rule: this is a Monday week-ahead brief, not a replay of Friday's session. "
                "Use week_ahead.key_events, week_ahead.calendar, and week_ahead.watch_items to lay out the week's "
                "calendar, base/risk case, and what to watch. Frame Friday's close as the baseline, not a fresh move. "
                "Do not re-review Friday's session in detail.\n"
            )
        else:
            non_trading_instruction = (
                "Non-trading-day rule: treat Hong Kong and A-share cash-market data as last-available reference tape only. "
                "Do not frame it as a fresh cash-session move. Focus analysis on still-moving financial actions: policy and regulatory signals, "
                "geopolitics, central-bank repricing, FX/commodities/crypto, corporate actions, and the next open. "
                "Use non_trading_focus.event_watch when supplied.\n"
            )

    common_requirements = (
        "Write in English only.\n"
        "Use only the supplied facts.\n"
        "Keep the tone calm, specific, and useful for a Hong Kong sell-side commute note.\n"
        "If data is missing, say so instead of filling gaps.\n"
        + non_trading_instruction
    )

    task_instructions = {
        "news_selection": (
            "Task: select the most important 3-5 overnight stories for a Hong Kong market reader.\n"
            "Order by likely impact on Hong Kong or offshore-China market thinking, not by headline drama.\n"
            "Return JSON with keys: summary, selected_news.\n"
            "selected_news items must include headline, why_it_matters, hk_market_impact, importance.\n"
        ),
        "overnight_review": (
            "Task: write the overnight overseas market review.\n"
            "Explain why markets moved, not just how much.\n"
            "Use the supplied deterministic driver decomposition as a consistency anchor; do not add causal claims that conflict with it.\n"
            "Use target_hk_session for today's open; next_hk_trading_day is the following session.\n"
            "Never print internal JSON keys or compare outperform/lag across different trading sessions; label prior-HK versus overnight-US evidence explicitly.\n"
            "Keep paragraph to 2-4 short sentences. Put detailed causal points in drivers rather than making one dense paragraph.\n"
            "End with what the move means for today's Hong Kong open or next Hong Kong session via hk_open_implication.\n"
            "Return JSON with keys: paragraph, drivers, hk_open_implication.\n"
        ),
        "hk_review": (
            "Task: write a Hong Kong / A-share review setup.\n"
            "Focus on style leadership, local flow implications, and cross-market read-through.\n"
            "Use the supplied flow evidence and deterministic driver decomposition when available; if Connect or CCASS data is absent, state that confirmation is incomplete.\n"
            "Do not rank prior-HK cash moves against overnight US-listed moves as outperform/lag; describe confirmation or reversal instead.\n"
            "The local_leadership field must include the style call, at least two supplied facts, and what the evidence means for the open; never return only a generic label such as 'Hong Kong growth / internet led'.\n"
            "Keep paragraph to 2-4 short sentences. Put the actionable confirmation test in follow_through.\n"
            "Return JSON with keys: paragraph, local_leadership, follow_through.\n"
        ),
        "macro_interpretation": (
            "Task: interpret the macro and policy agenda for today's report.\n"
            "Focus on what can reprice Hong Kong risk appetite, rates, FX, and sector leadership.\n"
            "Return JSON with keys: paragraph, watchpoints.\n"
        ),
        "company_commentary": (
            "Task: summarize the most relevant company and sector events.\n"
            "Point out beat/miss/inline logic, rating-change implications, and what matters for Hong Kong peers.\n"
            "Return JSON with keys: paragraph, company_notes.\n"
            "company_notes items must include ticker and commentary.\n"
        ),
        "theme_deep_dive": (
            "Task: expand the rotating weekly theme into a short desk-style note.\n"
            "Connect the theme angle, current signals, related names, and near-term catalysts.\n"
            "Return JSON with keys: paragraph, watch_items.\n"
        ),
        "final_framing": (
            "Task: write the final framing for the top and bottom of the report.\n"
            "Return JSON with keys: one_line_market_pulse, thinking_note, risk_check, interview_answer.\n"
            "The market pulse must be one sentence. The interview answer must be two short sentences.\n"
            "Keep the framing aligned with the supplied deterministic driver decomposition and flow evidence. Never print internal JSON field names.\n"
        ),
    }

    return (
        common_requirements
        + task_instructions.get(task_name, "")
        + "\n"
        + TASK_FEW_SHOTS.get(task_name, "")
        + "\n\nContext JSON:\n"
        + json.dumps(context, ensure_ascii=False)
    )


def _run_json_task_factory(llm_config: Dict[str, Any], cache_dir: str) -> TaskRunner:
    provider_semaphores = {
        str(provider): threading.Semaphore(max(int(cap), 1))
        for provider, cap in ((llm_config.get("provider_parallelism", {}) or {}).items())
        if str(cap).isdigit()
    }

    def runner(task_name: str, context: Dict[str, Any], prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        task_config = _task_config(llm_config, task_name)
        retries = max(int(llm_config.get("max_retries", 2)), 0) + 1
        temperature = float(task_config.get("temperature", 0.2))
        max_tokens = int(task_config.get("max_tokens", 700))
        last_error = ""
        last_usage: Dict[str, Any] = {}
        last_raw_excerpt = ""
        last_model = ""
        last_provider = ""
        last_route = ""
        total_attempts = 0

        for provider, model, route_name in _model_candidates(llm_config, task_name):
            last_model = model
            last_provider = provider
            last_route = route_name
            cached = _load_cache(cache_dir, task_name, context, model, prompt)
            if cached is not None:
                coerced_cached = _coerce_task_payload(task_name, cached)
                if _payload_has_content(coerced_cached):
                    return coerced_cached, {
                        "status": "cached",
                        "model": model,
                        "provider": provider,
                        "route": route_name,
                        "attempts": 0,
                    }

            # Retrying a truncated response at the same budget just truncates
            # again, so the allowance is raised for the retry instead. Measured
            # usage shows reasoning can consume the entire budget before any JSON
            # is emitted.
            attempt_tokens = max_tokens
            for attempt in range(1, retries + 1):
                total_attempts += 1
                try:
                    client = get_client(provider)
                    guard = provider_semaphores.get(provider)
                    with guard if guard is not None else nullcontext():
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=temperature,
                            max_tokens=attempt_tokens,
                            extra_body=get_completion_extra_body(provider, model),
                        )
                    raw = _extract_response_text(response.choices[0].message.content)
                    last_raw_excerpt = raw[:240].replace("\n", " ").strip()
                    finish_reason = _choice_finish_reason(response)
                    last_usage = _token_usage(response, attempt_tokens)
                    if _llm_response_looks_truncated(raw, finish_reason):
                        reason = finish_reason or "trailing ellipsis or unbalanced JSON"
                        # Give the next attempt more room rather than repeating
                        # a request that already ran out of it.
                        attempt_tokens = min(int(attempt_tokens * TRUNCATION_RETRY_MULTIPLIER), MAX_TOKENS_CEILING)
                        raise ValueError(
                            f"LLM response appears truncated ({reason}); retrying with a larger budget."
                        )
                    parsed = _extract_json_object(raw)
                    coerced = _coerce_task_payload(task_name, parsed)
                    if not _payload_has_content(coerced):
                        excerpt = f" Raw excerpt: {last_raw_excerpt}" if last_raw_excerpt else ""
                        raise ValueError(f"LLM returned an empty or non-parseable structured payload.{excerpt}")
                    _save_cache(cache_dir, task_name, context, model, prompt, coerced)
                    status = "fallback_ok" if route_name.endswith(":fallback") else "ok"
                    return coerced, {
                        "status": status,
                        "model": model,
                        "provider": provider,
                        "route": route_name,
                        "attempts": total_attempts,
                        "usage": last_usage,
                    }
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < retries and _is_retryable_error(last_error):
                        time.sleep(_retry_delay_seconds(llm_config, attempt, last_error))

        error_meta = {
            "status": "error",
            "model": last_model,
            "provider": last_provider,
            "route": last_route,
            "attempts": total_attempts,
            "error": last_error,
            "error_class": classify_error(last_error),
            "usage": last_usage,
        }
        if last_raw_excerpt:
            error_meta["raw_excerpt"] = last_raw_excerpt
        return TASK_DEFAULTS[task_name].copy(), error_meta

    return runner


def _parallel_phase_tasks(bundle: Dict[str, Any], llm_config: Dict[str, Any]) -> List[str]:
    tasks = ["news_selection", "macro_interpretation", "company_commentary", "theme_deep_dive"]
    return [task for task in tasks if _task_enabled(llm_config, task)]


def _task_condition(task_name: str, bundle: Dict[str, Any]) -> Tuple[bool, str]:
    if task_name == "news_selection":
        return _should_run_news_selection(bundle)
    if task_name == "macro_interpretation":
        return _should_run_macro(bundle)
    if task_name == "company_commentary":
        return _should_run_company(bundle)
    return True, ""


def _flatten_sections(task_outputs: Dict[str, Dict[str, Any]], task_meta: Dict[str, Any]) -> Dict[str, Any]:
    def sanitize(value: Any) -> Any:
        if isinstance(value, str):
            return re.sub(
                r"\battribution_v1\b",
                "the deterministic driver decomposition",
                value,
                flags=re.IGNORECASE,
            )
        if isinstance(value, dict):
            return {key: sanitize(child) for key, child in value.items()}
        if isinstance(value, list):
            return [sanitize(child) for child in value]
        return value

    task_outputs = sanitize(task_outputs)
    news_selection = task_outputs.get("news_selection", TASK_DEFAULTS["news_selection"])
    overnight_review = task_outputs.get("overnight_review", TASK_DEFAULTS["overnight_review"])
    hk_review = task_outputs.get("hk_review", TASK_DEFAULTS["hk_review"])
    macro_interpretation = task_outputs.get("macro_interpretation", TASK_DEFAULTS["macro_interpretation"])
    company_commentary = task_outputs.get("company_commentary", TASK_DEFAULTS["company_commentary"])
    theme_deep_dive = task_outputs.get("theme_deep_dive", TASK_DEFAULTS["theme_deep_dive"])
    final_framing = task_outputs.get("final_framing", TASK_DEFAULTS["final_framing"])

    return {
        "task_meta": task_meta,
        "tasks": task_outputs,
        "news_summary": news_selection.get("summary", ""),
        "selected_news": news_selection.get("selected_news", []),
        "deep_read_setup": overnight_review.get("paragraph", ""),
        "overnight_drivers": overnight_review.get("drivers", []),
        "overnight_hk_implication": overnight_review.get("hk_open_implication", ""),
        "hk_review_setup": hk_review.get("paragraph", ""),
        "hk_local_leadership": hk_review.get("local_leadership", ""),
        "hk_follow_through": hk_review.get("follow_through", ""),
        "macro_takeaway": macro_interpretation.get("paragraph", ""),
        "macro_watchpoints": macro_interpretation.get("watchpoints", []),
        "company_takeaway": company_commentary.get("paragraph", ""),
        "company_notes": company_commentary.get("company_notes", []),
        "theme_paragraph": theme_deep_dive.get("paragraph", ""),
        "theme_watch_items": theme_deep_dive.get("watch_items", []),
        "one_line_market_pulse": final_framing.get("one_line_market_pulse", ""),
        "thinking_note": final_framing.get("thinking_note", "") or theme_deep_dive.get("paragraph", ""),
        "risk_check": final_framing.get("risk_check", ""),
        "interview_answer": final_framing.get("interview_answer", ""),
    }


def generate_llm_sections(
    bundle: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    cache_dir: str = "",
    runner: Optional[TaskRunner] = None,
) -> Dict[str, Any]:
    llm_config = ((config or {}).get("llm", {}) or {})
    if not llm_config.get("enabled", True):
        return {"task_meta": {"status": "disabled", "reason": "LLM disabled in config."}}
    if not _api_key_present() and runner is None:
        return {"task_meta": {"status": "skipped", "reason": "No API key configured."}}

    task_outputs: Dict[str, Dict[str, Any]] = {task: TASK_DEFAULTS[task].copy() for task in TASK_ORDER}
    task_meta: Dict[str, Any] = {"status": "ok", "tasks": {}}
    runner_fn = runner or _run_json_task_factory(llm_config, cache_dir)

    # Phase 1: independent tasks.
    parallel_tasks = _parallel_phase_tasks(bundle, llm_config)
    max_workers = _effective_max_workers(llm_config, parallel_tasks)
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for task_name in parallel_tasks:
            should_run, reason = _task_condition(task_name, bundle)
            if not should_run:
                task_meta["tasks"][task_name] = {"status": "skipped", "reason": reason}
                continue
            context = _build_task_context(task_name, bundle, prior={})
            prompt = _build_prompt(task_name, context)
            futures[executor.submit(runner_fn, task_name, context, prompt)] = (task_name, context)

        for future in as_completed(futures):
            task_name, _ = futures[future]
            try:
                payload, meta = future.result()
                task_outputs[task_name] = _coerce_task_payload(task_name, payload)
                task_meta["tasks"][task_name] = meta
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                task_meta["tasks"][task_name] = {
                    "status": "error",
                    "error": message,
                    "error_class": classify_error(message),
                }

    # Phase 2: overnight review depends on curated news when available.
    if _task_enabled(llm_config, "overnight_review"):
        context = _build_task_context("overnight_review", bundle, prior=task_outputs)
        prompt = _build_prompt("overnight_review", context)
        payload, meta = runner_fn("overnight_review", context, prompt)
        task_outputs["overnight_review"] = _coerce_task_payload("overnight_review", payload)
        task_meta["tasks"]["overnight_review"] = meta
    else:
        task_meta["tasks"]["overnight_review"] = {"status": "skipped", "reason": "Task disabled in config."}

    # Phase 3: Hong Kong review depends on overnight framing plus local flow context.
    if _task_enabled(llm_config, "hk_review"):
        context = _build_task_context("hk_review", bundle, prior=task_outputs)
        prompt = _build_prompt("hk_review", context)
        payload, meta = runner_fn("hk_review", context, prompt)
        task_outputs["hk_review"] = _coerce_task_payload("hk_review", payload)
        task_meta["tasks"]["hk_review"] = meta
    else:
        task_meta["tasks"]["hk_review"] = {"status": "skipped", "reason": "Task disabled in config."}

    # Phase 4: final framing depends on all upstream outputs.
    if _task_enabled(llm_config, "final_framing"):
        context = _build_task_context("final_framing", bundle, prior=task_outputs)
        prompt = _build_prompt("final_framing", context)
        payload, meta = runner_fn("final_framing", context, prompt)
        task_outputs["final_framing"] = _coerce_task_payload("final_framing", payload)
        task_meta["tasks"]["final_framing"] = meta
    else:
        task_meta["tasks"]["final_framing"] = {"status": "skipped", "reason": "Task disabled in config."}

    for task_name in TASK_ORDER:
        task_meta["tasks"].setdefault(task_name, {"status": "skipped", "reason": "Task not run."})

    if any(meta.get("status") == "error" for meta in task_meta["tasks"].values()):
        task_meta["status"] = "partial"

    task_meta["health"] = build_llm_health(task_meta)

    return _flatten_sections(task_outputs, task_meta)


def build_llm_health(task_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise task outcomes so failure causes are countable across runs.

    Reporting only an error count made a sustained failure look like noise: the
    overlay ran at 0-2 of 7 successful tasks for fifteen consecutive days without
    the cause ever being visible in the archive.
    """
    tasks = (task_meta or {}).get("tasks", {}) or {}
    by_class: Dict[str, int] = {}
    by_task: Dict[str, str] = {}
    succeeded = 0
    failed = 0
    skipped = 0

    for name, meta in tasks.items():
        if not isinstance(meta, dict):
            continue
        status = str(meta.get("status", "")).lower()
        if status in {"ok", "cached"}:
            succeeded += 1
            continue
        if status == "skipped":
            skipped += 1
            continue
        if status == "error":
            failed += 1
            label = str(meta.get("error_class") or classify_error(str(meta.get("error", ""))))
            by_class[label] = by_class.get(label, 0) + 1
            by_task[name] = label

    attempted = succeeded + failed
    success_rate = round(succeeded / attempted * 100.0, 1) if attempted else 0.0
    dominant = max(by_class.items(), key=lambda pair: pair[1])[0] if by_class else ""

    # Budget pressure is now measured rather than inferred: a task that failed
    # while using nearly all of its allowance is a genuine budget problem, one
    # that failed using little of it is not.
    budget: Dict[str, Any] = {}
    for name, meta in tasks.items():
        if not isinstance(meta, dict):
            continue
        usage = meta.get("usage") or {}
        if isinstance(usage, dict) and usage.get("budget_used_pct") is not None:
            budget[name] = {
                "budget_used_pct": usage.get("budget_used_pct"),
                "completion_tokens": usage.get("completion_tokens"),
                "max_tokens": usage.get("max_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
            }

    return {
        "schema_version": "llm-health-v1",
        "tasks_total": len(tasks),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "success_rate_pct": success_rate,
        "failures_by_class": by_class,
        "failure_class_by_task": by_task,
        "dominant_failure_class": dominant,
        "token_budget_by_task": budget,
        "read": (
            f"{succeeded}/{attempted} tasks succeeded ({success_rate:.0f}%)."
            + (f" Dominant failure: {dominant} ({by_class.get(dominant, 0)} of {failed})." if dominant else "")
        ),
    }
