# Tests

Regression tests live here so the repository root stays focused on runtime entry points and documentation.

Run the full smoke suite from the project root:

```bash
python scripts/run_tests.py
```

The GitHub Actions workflow uses the same runner before generating the daily report.
Use `python scripts/run_tests.py --pytest` when you also want pytest collection.

## Coverage Map

- `test_analytics_*.py`: deterministic analytics modules, including macro, sector, flow, Hong Kong checks, watchlists, and narrative framing.
- `test_*_resilience.py`, `test_news_cache.py`, and adapter tests: data-source fallback and cache behavior.
- `test_professional_workbench.py`, `test_report_quality.py`, and rendering tests: end-to-end bundle/report output checks.
- `test_github_actions.py`: import and environment smoke checks that mirror CI startup risk.
