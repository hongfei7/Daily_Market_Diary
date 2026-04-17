# Troubleshooting

## Fast Checks

Run these commands first from the project root:

```bash
python test_github_actions.py
python test_professional_workbench.py
python market_diary/main_professional.py --date 2026-04-13 --no-llm
```

If those pass, the core data pipeline, template assembly, and chart dashboard are working.

## Common Issues

### Report generation fails before writing output

Typical causes:

- Python is missing or too old.
- Project dependencies are not installed.
- A live data source is temporarily unavailable.

Fix:

```bash
pip install -r market_diary/requirements.txt
python market_diary/main_professional.py --date 2026-04-13 --no-llm
```

Running with `--no-llm` helps separate data or chart issues from model issues.

### LLM request fails or times out

Typical causes:

- `MINIMAX_API_KEY` or `OPENAI_API_KEY` is missing.
- The upstream model endpoint is overloaded.
- The configured `LLM_BASE_URL` is wrong.

Fix:

```bash
set MINIMAX_API_KEY=your_key_here
python market_diary/main_professional.py --date 2026-04-13
```

If the model provider is unstable, keep the production run alive with:

```bash
python market_diary/main_professional.py --date 2026-04-13 --no-llm
```

The professional bundle and dashboard will still be generated.

### Data looks stale

This is usually expected on weekends, public holidays, or before a market session is open.

Behavior:

- The system falls back to the latest tradable session for intraday charts.
- The report metadata records both the requested date and the effective data date.

Check:

```bash
python market_diary/main_professional.py --date 2026-04-13 --no-llm
```

Then inspect the report header and the appendix metadata.

### Charts fail to render

Typical causes:

- `matplotlib` is missing.
- The environment does not have a working backend.

Fix:

```bash
pip install -r market_diary/requirements.txt --force-reinstall
```

### GitHub Actions cannot push reports

Typical causes:

- Workflow permissions are read-only.
- Branch protection blocks `github-actions[bot]`.

Fix:

1. In repository settings, set workflow permissions to `Read and write`.
2. Allow the bot to push to `main`, or exempt the workflow from the protection rule.

## Debugging Aids

### Validate the local professional stack

```bash
python test_professional_workbench.py
```

### Validate the CI smoke path

```bash
python test_github_actions.py
```

### Inspect a single adapter

```python
from market_diary.modules.macro_calendar import fetch_macro_data
print(fetch_macro_data("2026-04-13"))
```

### Verify the LLM client wiring

```python
from market_diary.modules.llm_client import get_client
print(get_client())
```

## Where To Look

- `README_PROFESSIONAL.md` for the overall system and repo layout.
- `docs/QUICK_START.md` for day-one setup.
- `docs/UPGRADE_GUIDE.md` for the student-to-professional migration path.
