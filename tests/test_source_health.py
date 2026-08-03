from __future__ import annotations

from market_diary.professional.source_health import build_source_health


def _record(source_type: str, status: str, as_of: str, confidence: float = 0.9) -> dict:
    return {
        "source_name": "Fixture",
        "source_url": "https://example.com",
        "as_of": as_of,
        "collected_at": f"{as_of}T01:00:00+00:00",
        "source_type": source_type,
        "status": status,
        "confidence": confidence,
    }


def test_source_health_separates_authority_freshness_and_availability() -> None:
    health = build_source_health(
        {
            "market_data": [_record("public", "ok", "2026-08-01")],
            "hk_local": [_record("official", "ok", "2026-07-31")],
            "sector_news": [_record("unavailable", "unavailable", "2026-08-03", 0.0)],
        },
        reference_date="2026-08-03",
        policies={
            "market_data": {"critical": True, "max_age_days": 4},
            "hk_local": {"critical": True, "max_age_days": 4},
            "sector_news": {"critical": False, "max_age_days": 3},
        },
    )
    assert health["status"] == "degraded"
    by_name = {row["source"]: row for row in health["sources"]}
    assert by_name["market_data"]["status"] == "healthy"
    assert by_name["hk_local"]["dimensions"]["authority"] == 100.0
    assert by_name["sector_news"]["status"] == "unavailable"
    assert not health["critical_failures"]


def test_future_dated_critical_source_fails() -> None:
    health = build_source_health(
        {"market_data": [_record("public", "ok", "2026-08-04")]},
        reference_date="2026-08-03",
        policies={"market_data": {"critical": True, "max_age_days": 4}},
    )
    assert health["status"] == "failed"
    assert "market_data" in health["critical_failures"]
    assert any("future" in warning for warning in health["warnings"])


if __name__ == "__main__":
    test_source_health_separates_authority_freshness_and_availability()
    test_future_dated_critical_source_fails()
    print("Source health tests passed")
