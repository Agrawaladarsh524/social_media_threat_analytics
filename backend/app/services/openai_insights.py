"""
Multi-prompt OpenAI generation: compact OSINT JSON -> structured UI payloads.
Each fragment is a separate model call (sequential for client stability).
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

BASE = """You are an OSINT analyst helping a user understand their public footprint.
Rules:
- Use ONLY information present in USER_JSON. Do not invent employers, DMs, private facts, or platforms not in USER_JSON.
- If USER_JSON is sparse, say so briefly and lower confidence in evidence strings.
- Return valid JSON only (no markdown fences)."""


def _trunc(s: str | None, n: int = 450) -> str:
    if not s:
        return ''
    return s if len(s) <= n else s[: n - 1] + '…'


def compact_datasets(data: dict[str, Any]) -> str:
    """Shrink payload for token limits while keeping cross-platform signals."""
    out: dict[str, Any] = {}

    tw = data.get('twitter') or []
    if isinstance(tw, list):
        compact_tw = []
        for t in tw[:40]:
            if not isinstance(t, dict): 
                continue
                
            # Extract Text dynamically
            text = t.get('text') or t.get('full_text') or t.get('tweet') or t.get('content')
            if not text:
                # Find longest text field as fallback (likely the tweet body)
                longest = ""
                for k, v in t.items():
                    if isinstance(v, str) and len(v) > len(longest) and 'url' not in k.lower() and 'link' not in k.lower():
                        longest = v
                text = longest

            # Extract Date
            date = t.get('createdAt') or t.get('created_at') or t.get('date') or t.get('timestamp')

            tw_obj = {
                'text': _trunc(str(text), 400) if text else '',
                'date': date
            }
            compact_tw.append(tw_obj)
            
        out['twitter'] = compact_tw

    return json.dumps(out, ensure_ascii=False)


def _generate_fragment(
    client: OpenAI,
    model: str,
    schema: str,
    user_json: str,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': BASE},
            {'role': 'user', 'content': f'JSON_SCHEMA:\n{schema}\n\nUSER_JSON:\n{user_json}'},
        ],
        response_format={'type': 'json_object'},
        temperature=0.2,
    )
    text = (response.choices[0].message.content or '').strip()
    return json.loads(text)


SCHEMA_EXECUTIVE = """{
  "riskScore": <integer 0-100>,
  "riskMax": 100,
  "riskLabel": <short string>,
  "summary": <2-4 sentences string>
}"""

SCHEMA_INFERENCES = """{
  "rows": [
    {
      "targetInfo": <string category>,
      "inferredValue": <string>,
      "evidence": <string ending with (High) or (Medium) or (Low) in parentheses>
    }
  ]
}
Produce 4-8 rows where possible from USER_JSON only."""

SCHEMA_PATTERN = """{
  "entries": [
    {
      "id": <string "1","2",...>,
      "place": <string>,
      "period": <optional string or omit for hometown>,
      "detail": <string>,
      "source": <string citing which field/post in USER_JSON>
    }
  ],
  "polInference": <one sentence attacker takeaway>
}"""

SCHEMA_PHISHING = """{
  "sims": [
    {
      "id": <slug string>,
      "title": <string>,
      "from": <spoofed from line>,
      "to": <target display name>,
      "subject": <string>,
      "body": <string multiline plausible phish body, no real links>,
      "why": <string explaining which public signal was abused>
    }
  ]
}
Produce exactly 2 simulations."""

SCHEMA_METHODOLOGY = """{
  "headline": <string>,
  "intro": <2-4 sentences>,
  "pillars": [
    {"title": <string>, "subtitle": <string>, "text": <string>},
    {"title": <string>, "subtitle": <string>, "text": <string>},
    {"title": <string>, "subtitle": <string>, "text": <string>}
  ]
}"""


def generate_ai_bundle(
    api_key: str,
    model_name: str,
    data: dict[str, Any],
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Run five OpenAI requests and merge. Raises on hard failure.
    `timeout` is the per-request timeout in seconds.
    """
    client = OpenAI(api_key=api_key, timeout=timeout)
    user_block = compact_datasets(data)

    executive = _generate_fragment(client, model_name, SCHEMA_EXECUTIVE, user_block)
    inferences = _generate_fragment(client, model_name, SCHEMA_INFERENCES, user_block)
    pattern = _generate_fragment(client, model_name, SCHEMA_PATTERN, user_block)
    phishing = _generate_fragment(client, model_name, SCHEMA_PHISHING, user_block)
    methodology = _generate_fragment(client, model_name, SCHEMA_METHODOLOGY, user_block)

    rs = executive.get('riskScore')
    if not isinstance(rs, int):
        try:
            executive['riskScore'] = int(float(rs))
        except (TypeError, ValueError):
            executive['riskScore'] = 50
    executive.setdefault('riskMax', 100)
    executive.setdefault('riskLabel', 'Assessed from public footprint')
    executive.setdefault('summary', '')

    rows = inferences.get('rows')
    if not isinstance(rows, list):
        rows = []

    entries = pattern.get('entries')
    if not isinstance(entries, list):
        entries = []
    pol_inf = pattern.get('polInference')
    if not isinstance(pol_inf, str):
        pol_inf = str(pol_inf or '')

    sims = phishing.get('sims')
    if not isinstance(sims, list):
        sims = []

    if not isinstance(methodology, dict):
        methodology = {}
    methodology.setdefault('headline', 'Methodology')
    methodology.setdefault('intro', '')
    methodology.setdefault('pillars', [])

    return {
        'executive': executive,
        'inferenceRows': rows,
        'patternOfLife': entries,
        'polInference': pol_inf,
        'phishingSims': sims,
        'methodology': methodology,
    }
