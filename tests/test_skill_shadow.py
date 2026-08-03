from __future__ import annotations

from _bootstrap import ROOT

from professional.config import load_professional_config
from professional.skill_shadow import SKILL_NAMES, generate_skill_shadow


def _bundle():
    return {
        "meta": {"report_date": "2026-08-03"},
        "date_semantics": {"lines": ["Global data through 2026-08-02"]},
        "provenance_audit": {"status": "ok"},
        "overview": {
            "theme": "Lower volatility but higher yields",
            "risk_regime": "Mixed",
            "snapshot_rows": [
                {"label": "S&P 500", "change_pct": 0.7},
                {"label": "US 10Y", "change_pct": 1.2},
            ],
        },
        "hk_quick_checks": [],
        "attribution": {},
        "must_watch": [{"title": "US 10Y higher", "summary": "Valuation pressure remains."}],
        "today_forward": {"focus_lines": ["Check HSTECH breadth."]},
        "llm_sections": {"one_line_market_pulse": "Mixed external setup."},
        "macro_agenda": [],
        "company_events": {"earnings": [], "ratings": [], "announcements": []},
        "watchlists": {
            "Core coverage": [
                {
                    "ticker": "0700.HK",
                    "name": "Tencent",
                    "thesis": "Advertising and gaming support cash flow.",
                    "upcoming_catalyst": "",
                }
            ]
        },
        "sector_digest": {"graded_news": []},
    }


def test_skill_files_are_project_local_and_complete() -> None:
    for skill_name in SKILL_NAMES:
        skill_path = ROOT / "skills" / skill_name / "SKILL.md"
        schema_path = ROOT / "skills" / skill_name / "references" / "output-contract.md"
        assert skill_path.exists()
        assert schema_path.exists()
        assert "TODO" not in skill_path.read_text(encoding="utf-8")


def test_shadow_run_is_non_publishing_and_provider_agnostic() -> None:
    seen = []

    def fake_runner(skill_name, context, prompt, provider, model):
        seen.append(skill_name)
        assert provider == "deepseek"
        assert model == "deepseek-v4-pro"
        assert "strict JSON" in prompt
        assert "Verified context JSON" in prompt
        assert "claude" not in provider.lower()
        return {"gaps": [], "skill": skill_name}, {"status": "ok", "provider": provider, "model": model}

    config = load_professional_config()
    result = generate_skill_shadow(_bundle(), config=config, runner=fake_runner)

    assert seen == list(SKILL_NAMES)
    assert result["status"] == "ok"
    assert result["mode"] == "shadow"
    assert result["publish"] is False
    assert result["human_review_required"] is True
    assert set(result["skills"]) == set(SKILL_NAMES)


def main() -> None:
    test_skill_files_are_project_local_and_complete()
    test_shadow_run_is_non_publishing_and_provider_agnostic()
    print("Skill shadow test passed")


if __name__ == "__main__":
    main()
