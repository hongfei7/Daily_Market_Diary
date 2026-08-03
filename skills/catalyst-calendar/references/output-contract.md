# Output contract

Return one JSON object:

```json
{
  "events": [
    {
      "date": "YYYY-MM-DD",
      "event": "",
      "type": "macro|earnings|policy|corporate|industry",
      "impact": "high|medium|low",
      "affected_assets": [""],
      "why_it_matters": "",
      "upside_path": "",
      "downside_path": "",
      "confirmation": "",
      "invalidation": "",
      "research_question": "",
      "source_url": "",
      "as_of": ""
    }
  ],
  "undated_watch": [""],
  "gaps": [""]
}
```

Return no prose outside JSON. Limit `events` to eight verified items.
