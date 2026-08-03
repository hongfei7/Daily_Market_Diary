from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from market_diary.modules.llm_client import (
    get_available_providers,
    get_client,
    get_completion_extra_body,
    get_default_model,
)
from market_diary.professional.llm_enhancer import (
    _choice_finish_reason,
    _extract_json_object,
    _extract_response_text,
    _llm_response_looks_truncated,
)


SKILL_NAMES = ("morning-note", "catalyst-calendar", "thesis-tracker", "report-evidence-qc")
REQUIRED_OUTPUT_KEYS = {
    "morning-note": {"top_call", "signal_stack", "content_budget", "gaps"},
    "catalyst-calendar": {"events", "undated_watch", "gaps"},
    "thesis-tracker": {"updates", "portfolio_level_gaps", "gaps"},
    "report-evidence-qc": {
        "release_state",
        "release_reason",
        "claim_checks",
        "visual_checks",
        "priority_fixes",
        "caveats_to_publish",
        "gaps",
    },
}
ShadowRunner = Callable[[str, Dict[str, Any], str, str, str], Tuple[Dict[str, Any], Dict[str, Any]]]


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[2] / "skills"


def _load_skill_text(skill_name: str) -> str:
    skill_dir = _skills_root() / skill_name
    parts = [(skill_dir / "SKILL.md").read_text(encoding="utf-8")]
    for reference in sorted((skill_dir / "references").glob("*.md")):
        if reference.name.startswith("._"):
            continue
        parts.append(reference.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def _skill_version(skill_name: str) -> str:
    return hashlib.sha256(_load_skill_text(skill_name).encode("utf-8")).hexdigest()[:12]


def _watchlist_rows(bundle: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for bucket, items in (bundle.get("watchlists", {}) or {}).items():
        for item in (items or [])[:4]:
            rows.append(
                {
                    "bucket": bucket,
                    "ticker": item.get("ticker", ""),
                    "name": item.get("name", ""),
                    "thesis": item.get("thesis", ""),
                    "upcoming_catalyst": item.get("upcoming_catalyst", ""),
                    "stories": (item.get("stories", []) or [])[:2],
                }
            )
    return rows[:10]


def _shadow_context(skill_name: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    common = {
        "meta": bundle.get("meta", {}) or {},
        "date_semantics": bundle.get("date_semantics", {}) or {},
        "provenance_audit": bundle.get("provenance_audit", {}) or {},
    }
    if skill_name == "morning-note":
        common.update(
            {
                "overview": bundle.get("overview", {}) or {},
                "hk_quick_checks": (bundle.get("hk_quick_checks", []) or [])[:8],
                "attribution": bundle.get("attribution", {}) or {},
                "must_watch": (bundle.get("must_watch", []) or [])[:8],
                "today_forward": bundle.get("today_forward", {}) or {},
                "production_llm_sections": bundle.get("llm_sections", {}) or {},
            }
        )
    elif skill_name == "catalyst-calendar":
        common.update(
            {
                "macro_agenda": (bundle.get("macro_agenda", []) or [])[:12],
                "company_events": bundle.get("company_events", {}) or {},
                "today_forward": bundle.get("today_forward", {}) or {},
                "watchlists": _watchlist_rows(bundle),
            }
        )
    elif skill_name == "thesis-tracker":
        common.update(
            {
                "watchlists": _watchlist_rows(bundle),
                "company_events": bundle.get("company_events", {}) or {},
                "sector_digest": bundle.get("sector_digest", {}) or {},
                "today_forward": bundle.get("today_forward", {}) or {},
            }
        )
    else:
        common.update(
            {
                "overview": bundle.get("overview", {}) or {},
                "market_summary": bundle.get("market_summary", {}) or {},
                "hk_quick_checks": (bundle.get("hk_quick_checks", []) or [])[:10],
                "attribution": bundle.get("attribution", {}) or {},
                "daily_one_chart": bundle.get("daily_one_chart", {}) or {},
                "report_quality": bundle.get("report_quality", {}) or {},
                "fact_check": bundle.get("fact_check", {}) or {},
                "source_health": bundle.get("source_health", {}) or {},
                "llm_sections": bundle.get("llm_sections", {}) or {},
            }
        )
    return common


def _build_shadow_prompt(skill_name: str, context: Dict[str, Any]) -> str:
    return (
        "Execute the following provider-agnostic financial research skill in shadow mode. "
        "The output is for human comparison only and must not contain investment advice. "
        "Use only supplied facts and return strict JSON. Keep the whole answer under 700 words, "
        "prioritize the three most material findings, and omit generic background.\n\n"
        + _load_skill_text(skill_name)
        + "\n\nVerified context JSON:\n"
        + json.dumps(context, ensure_ascii=False, default=str)
    )


def _cache_path(cache_dir: str, skill_name: str, provider: str, model: str, prompt: str) -> Path:
    digest = hashlib.sha256(f"{provider}|{model}|{prompt}".encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"shadow_{skill_name}_{digest}.json"


def _output_contract_errors(skill_name: str, payload: Dict[str, Any]) -> list[str]:
    required = REQUIRED_OUTPUT_KEYS.get(skill_name, set())
    missing = sorted(key for key in required if key not in payload)
    errors = [f"Missing required output key: {key}" for key in missing]
    if skill_name == "report-evidence-qc" and payload.get("release_state") not in {
        "ready",
        "share_with_caveats",
        "needs_revision",
    }:
        errors.append("release_state is outside the output contract.")
    return errors


def _run_shadow_factory(shadow_config: Dict[str, Any], cache_dir: str) -> ShadowRunner:
    def runner(
        skill_name: str,
        context: Dict[str, Any],
        prompt: str,
        provider: str,
        model: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        path = _cache_path(cache_dir, skill_name, provider, model, prompt) if cache_dir else None
        if path and path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached:
                    return cached, {"status": "cached", "provider": provider, "model": model}
            except (OSError, json.JSONDecodeError):
                pass

        try:
            response = get_client(provider).chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an institutional research QA analyst. Use only verified context and return strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=float(shadow_config.get("temperature", 0.0)),
                max_tokens=int(shadow_config.get("max_tokens", 1800)),
                extra_body=get_completion_extra_body(provider, model),
            )
            raw = _extract_response_text(response.choices[0].message.content)
            if _llm_response_looks_truncated(raw, _choice_finish_reason(response)):
                raise ValueError("Shadow response was truncated.")
            parsed = _extract_json_object(raw)
            if not parsed:
                raise ValueError("Shadow response did not contain a JSON object.")
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            return parsed, {"status": "ok", "provider": provider, "model": model}
        except Exception as exc:
            return {}, {
                "status": "error",
                "provider": provider,
                "model": model,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return runner


def generate_skill_shadow(
    bundle: Dict[str, Any],
    config: Dict[str, Any] | None = None,
    cache_dir: str = "",
    runner: ShadowRunner | None = None,
) -> Dict[str, Any]:
    llm_config = ((config or {}).get("llm", {}) or {})
    shadow_config = (llm_config.get("skill_shadow", {}) or {})
    env_enabled = (os.getenv("DMD_SKILL_SHADOW_ENABLED") or "").strip().lower()
    enabled = bool(shadow_config.get("enabled", True))
    if env_enabled in {"0", "false", "no", "off"}:
        enabled = False
    elif env_enabled in {"1", "true", "yes", "on"}:
        enabled = True
    if not enabled:
        return {"status": "disabled", "mode": "shadow", "publish": False, "skills": {}}

    cadence = str(shadow_config.get("cadence", "weekly") or "weekly").lower()
    force_run = (os.getenv("DMD_SKILL_SHADOW_FORCE") or "").strip().lower() in {"1", "true", "yes", "on"}
    day_mode = str((bundle.get("day_mode", {}) or {}).get("mode", "") or "")
    if cadence == "weekly" and day_mode != "weekly_review" and not force_run:
        return {
            "status": "skipped",
            "mode": "shadow",
            "publish": False,
            "cadence": cadence,
            "reason": "Weekly research shadow is reserved for the weekly review to protect the daily delivery SLA and token budget.",
            "skills": {},
        }

    provider = str(shadow_config.get("provider", "deepseek") or "deepseek").strip().lower()
    if runner is None and provider not in get_available_providers():
        return {
            "status": "skipped",
            "mode": "shadow",
            "publish": False,
            "provider": provider,
            "reason": f"{provider} API key is not configured.",
            "skills": {},
        }

    model_env = str(shadow_config.get("model_env", "DMD_SKILL_SHADOW_MODEL") or "")
    model = (os.getenv(model_env) or "").strip() if model_env else ""
    model = model or get_default_model(provider)
    runner_fn = runner or _run_shadow_factory(shadow_config, cache_dir)
    results: Dict[str, Any] = {}
    for skill_name in SKILL_NAMES:
        context = _shadow_context(skill_name, bundle)
        prompt = _build_shadow_prompt(skill_name, context)
        payload, meta = runner_fn(skill_name, context, prompt, provider, model)
        contract_errors = _output_contract_errors(skill_name, payload) if payload else []
        if str(meta.get("status", "")) in {"ok", "cached"} and contract_errors:
            meta = {
                **meta,
                "status": "error",
                "error": "Output contract validation failed.",
                "contract_errors": contract_errors,
            }
        results[skill_name] = {
            "skill_version": _skill_version(skill_name),
            "output": payload,
            "meta": meta,
        }

    statuses = [str((item.get("meta", {}) or {}).get("status", "")) for item in results.values()]
    status = "ok" if statuses and all(value in {"ok", "cached"} for value in statuses) else "partial"
    return {
        "status": status,
        "mode": "shadow",
        "publish": False,
        "provider": provider,
        "model": model,
        "human_review_required": True,
        "skills": results,
    }
