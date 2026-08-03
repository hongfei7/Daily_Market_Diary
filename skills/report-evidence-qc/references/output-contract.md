# Output contract

Return one JSON object:

```json
{
  "release_state": "ready|share_with_caveats|needs_revision",
  "release_reason": "",
  "claim_checks": [
    {
      "claim": "",
      "status": "verified|bounded_interpretation|unsupported|conflicted",
      "evidence": [""],
      "source_url": "",
      "as_of": "",
      "issue": "",
      "required_action": ""
    }
  ],
  "visual_checks": [
    {
      "visual": "",
      "status": "pass|caveat|fail",
      "takeaway_supported": true,
      "issue": "",
      "required_action": ""
    }
  ],
  "priority_fixes": [""],
  "caveats_to_publish": [""],
  "gaps": [""]
}
```

Return no prose outside JSON. Use empty arrays when no item is present.
