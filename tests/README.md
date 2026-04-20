# Tests

Regression tests live here so the repository root stays focused on runtime entry points and documentation.

Run the full smoke suite from the project root:

```bash
python tests/test_github_actions.py
python tests/test_professional_workbench.py
```

The GitHub Actions workflow runs the broader script-based suite before generating the daily report.
