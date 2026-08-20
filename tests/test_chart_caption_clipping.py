"""Chart captions must never lose their tail silently.

Both chart modules clipped a wrapped caption with ``textwrap.shorten``, which
only appends its placeholder when the text exceeds the width. A wrapped line is
already under the width by construction, so the placeholder never appeared and
readers saw sentences that simply stopped:

    "Local evidence only; unavailable fields stay out of the"
    "If watchlist names dominate, the desk should prepare for"

This is the same defect class as the prose fragments in the markdown, which
``prose_guard`` now catches. ``prose_guard`` cannot see inside a PNG, so these
tests guard the chart side.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import pytest

from market_diary.professional.daily_one_chart import _wrap_text as one_chart_wrap
from market_diary.professional.dashboard import _wrap_text as dashboard_wrap

# The two captions that shipped truncated.
REGRESSION_CAPTIONS = [
    "Local evidence only; unavailable fields stay out of the signal",
    "If watchlist names dominate, the desk should prepare for stock-specific follow-up questions.",
]


@pytest.mark.parametrize("wrap", [dashboard_wrap, one_chart_wrap], ids=["dashboard", "daily_one_chart"])
@pytest.mark.parametrize("caption", REGRESSION_CAPTIONS)
def test_clipped_caption_is_marked(wrap, caption):
    out = wrap(caption, 58, 1)
    assert out != caption, "caption should have been clipped at this width"
    assert out.endswith("…"), f"clip happened without a mark: {out!r}"


@pytest.mark.parametrize("wrap", [dashboard_wrap, one_chart_wrap], ids=["dashboard", "daily_one_chart"])
def test_caption_that_fits_is_untouched(wrap):
    caption = "Comparable 1D returns; colors reflect HK decision impact"
    assert wrap(caption, 58, 1) == caption


@pytest.mark.parametrize("wrap", [dashboard_wrap, one_chart_wrap], ids=["dashboard", "daily_one_chart"])
def test_clip_mark_respects_the_width_budget(wrap):
    caption = "word " * 40
    for width in (20, 32, 58):
        out = wrap(caption, width, 1)
        assert all(len(line) <= width for line in out.split("\n")), out
        assert out.endswith("…")


@pytest.mark.parametrize("wrap", [dashboard_wrap, one_chart_wrap], ids=["dashboard", "daily_one_chart"])
def test_empty_input_is_safe(wrap):
    assert wrap("", 58, 1) == ""
    assert wrap(None, 58, 1) == ""


@pytest.mark.parametrize("wrap", [dashboard_wrap, one_chart_wrap], ids=["dashboard", "daily_one_chart"])
def test_multi_line_budget_still_marks_the_last_line(wrap):
    caption = (
        "Read concentration before the headline aggregate; a narrow flow is not broad confirmation "
        "and the desk should treat it with more caution than the headline number implies."
    )
    out = wrap(caption, 48, 2)
    assert len(out.split("\n")) == 2
    assert out.endswith("…")


def test_source_captions_now_fit_their_budget():
    """The two shipped offenders were shortened at source, not just marked."""
    from market_diary.professional import dashboard, daily_one_chart

    src = open(dashboard.__file__, encoding="utf-8").read()
    assert "unavailable fields stay out of the signal" not in src

    src = open(daily_one_chart.__file__, encoding="utf-8").read()
    assert "the desk should prepare for stock-specific follow-up questions" not in src
