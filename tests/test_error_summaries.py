from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from modules.local_metrics import append_error_record, summarize_error_records


def test_summarize_error_records_compacts_repeated_contexts() -> None:
    errors = []
    append_error_record(errors, source="HKEX Daily Quotations", message="timed out", error_type="ReadTimeout", context="2026-04-14")
    append_error_record(errors, source="HKEX Daily Quotations", message="timed out", error_type="ReadTimeout", context="2026-04-13")
    append_error_record(errors, source="HKEX Daily Quotations", message="timed out", error_type="ReadTimeout", context="2026-04-12")
    append_error_record(errors, source="HKMA", message="bad json", error_type="ValueError", context="2026-04-14 invalid-json")

    summary = summarize_error_records(errors, limit=10)

    assert len(summary) == 2
    assert "HKEX Daily Quotations: ReadTimeout: timed out (x3;" in summary[0]
    assert "2026-04-14" in summary[0]
    assert "HKMA [2026-04-14 invalid-json]: ValueError: bad json" == summary[1]
