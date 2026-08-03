# Output contract

Return one JSON object:

```json
{
  "updates": [
    {
      "ticker": "",
      "thesis": "",
      "assessment": "strengthens|weakens|mixed|neutral|insufficient_evidence",
      "confirming_evidence": [""],
      "disconfirming_evidence": [""],
      "next_decisive_observation": "",
      "invalidation": "",
      "human_follow_up": "",
      "source_url": "",
      "as_of": ""
    }
  ],
  "portfolio_level_gaps": [""],
  "gaps": [""]
}
```

Return no prose outside JSON. Include only supplied coverage names and limit `updates` to eight items.
