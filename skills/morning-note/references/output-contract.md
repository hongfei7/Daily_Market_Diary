# Output contract

Return one JSON object:

```json
{
  "top_call": {
    "claim": "",
    "evidence": [""],
    "why_it_matters": "",
    "hk_implication": "",
    "confirmation": "",
    "invalidation": "",
    "confidence": "high|medium|low"
  },
  "signal_stack": [
    {"signal": "", "move": "", "interpretation": "", "next_check": "", "source_url": "", "as_of": ""}
  ],
  "content_budget": {"scan": [""], "deep_read": [""], "appendix": [""]},
  "gaps": [""]
}
```

Return no prose outside JSON. Limit `signal_stack` to four items and each content-budget list to five items.
