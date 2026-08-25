"""Content-quality guards found by auditing the rendered report.

Each test here corresponds to something the report actually said that was wrong,
redundant, or unverifiable — not to a hypothetical.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import pytest

from market_diary.professional.analytics_macro import _macro_profile
from market_diary.professional.attribution import build_attribution
from market_diary.professional.config import load_professional_config


def _item(pct):
    return {"Price": 100.0, "Pct Change": pct, "Trading Freshness Days": 1,
            "Change Value": pct, "Change Unit": "pct"}


class TestSemisEnterTheRiskScore:
    """The score said "Risk-on 72.2" while SOXX fell 2.21% and SMIC 3.76%.

    The composite had no semiconductor term at all, so it contradicted the chart
    printed directly beneath it.
    """

    def _summary(self, with_soxx: bool):
        equities = {
            "S&P 500": _item(0.21),
            "Hang Seng TECH ETF": _item(-1.02),
            "China Large-Cap (FXI)": _item(1.77),
        }
        if with_soxx:
            equities["Semiconductors (SOXX)"] = _item(-2.21)
        return {"Equities": equities, "FX": {"DXY": _item(-0.82)}, "Vol": {"VIX": _item(-6.00)}}

    def test_a_semis_selloff_pulls_the_score_down(self):
        without = build_attribution(self._summary(False), {}, {}, {})["risk_dashboard"]
        with_semis = build_attribution(self._summary(True), {}, {}, {})["risk_dashboard"]
        assert with_semis["score"] < without["score"]
        assert with_semis["bucket"] != "Risk-on"

    def test_the_component_is_named_with_its_evidence(self):
        rd = build_attribution(self._summary(True), {}, {}, {})["risk_dashboard"]
        semis = next(c for c in rd["components"] if c["label"] == "Semis complex")
        assert "SOXX" in semis["evidence"]

    def test_absent_semis_coverage_does_not_invent_a_component(self):
        rd = build_attribution(self._summary(False), {}, {}, {})["risk_dashboard"]
        assert not any(c["label"] == "Semis complex" for c in rd["components"])


class TestMacroProfileIsCountryAware:
    """Hong Kong CPI was described as moving US rates and the dollar.

    Substring matching hit the generic (US) CPI profile. The Hong Kong dollar is
    pegged, so local CPI does not drive US yields.
    """

    def setup_method(self):
        self.config = load_professional_config()

    def test_hong_kong_cpi_gets_its_own_transmission(self):
        profile = _macro_profile("Hong Kong CPI", self.config, "HK")
        assert "dollar" not in profile["impact"].lower()
        assert "funding" in profile["impact"].lower()

    def test_china_cpi_gets_its_own_transmission(self):
        profile = _macro_profile("China CPI / PPI", self.config, "CN")
        assert "China demand" in profile["impact"]

    def test_us_cpi_keeps_the_rates_and_dollar_read(self):
        profile = _macro_profile("US CPI", self.config, "US")
        assert "dollar" in profile["impact"].lower()

    def test_country_qualified_keys_do_not_leak_without_a_country(self):
        """"HK CPI" must not match a US print just because it contains "CPI"."""
        profile = _macro_profile("US CPI", self.config, "")
        assert "dollar" in profile["impact"].lower()

    def test_unknown_indicator_still_degrades_to_the_generic_profile(self):
        profile = _macro_profile("Some Unlisted Release", self.config, "HK")
        assert profile["industries"] == ["Market-wide"]


class TestDivergencePoolIsRelevant:
    """"The biggest cross-asset divergence was Bitcoin versus Oil".

    The candidate pool held only Gold, Oil and Bitcoin, so the headline was
    always a comparison among those three — and it drove the report theme, the
    first checklist item and the dashboard subtitle.
    """

    def test_only_same_session_hong_kong_proxies_are_candidates(self):
        from market_diary.modules.chart_features import _ASSET_KEYWORDS, _DIVERGENCE_CANDIDATES

        assert {"Nasdaq 100", "Hang Seng", "3033.HK", "FXI"} <= set(_ASSET_KEYWORDS)
        assert _DIVERGENCE_CANDIDATES == {"Nasdaq 100", "FXI"}

    def test_commodities_and_crypto_cannot_be_the_headline_on_their_own(self):
        from market_diary.modules.chart_features import _DIVERGENCE_CANDIDATES

        assert not _DIVERGENCE_CANDIDATES & {"Gold", "Oil", "Bitcoin"}


def test_asset_rows_explain_rather_than_restate():
    """"+0.21%" became "US beta improved" — a translation, not an explanation."""
    from market_diary.professional.report_sections import _asset_interpretation

    bundle = {
        "market_summary": {
            "Equities": {"S&P 500": _item(0.43), "Nasdaq 100": _item(0.33)},
            "Vol": {"VIX": _item(-5.50)},
        }
    }
    interpretation, check = _asset_interpretation(bundle, "S&P 500", 0.43)
    # Cites the driver rather than restating the sign.
    assert "volatility" in interpretation.lower()
    assert not check.startswith("Confirm with")


def test_missing_driver_is_admitted_not_invented():
    from market_diary.professional.report_sections import _asset_interpretation

    bundle = {"market_summary": {"Equities": {"S&P 500": _item(0.43)}}}
    interpretation, _ = _asset_interpretation(bundle, "S&P 500", 0.43)
    assert "without a clear driver" in interpretation


@pytest.mark.parametrize("threshold_case", [
    ([(None, -2.0, "unused"), (-3.0, -2.0, "picked")], "picked"),
    ([(0.0, -2.0, "unused")], ""),
    ([], ""),
])
def test_driver_selection_is_honest_about_nothing_matching(threshold_case):
    from market_diary.professional.report_sections import _attributed_driver

    candidates, expected = threshold_case
    assert _attributed_driver(candidates) == expected
