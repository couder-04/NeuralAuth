"""
output_schema.py
================
The JSON response contract the LLM must follow. Kept separate so
validator.py and prompts.py can both reference the same wording instead
of two independently-drifting copies.
"""


def response_format_block(allowed_fields: str) -> str:
    return f"""\
Respond with ONLY a JSON array -- no prose, no markdown fences, no text outside the array.
Each element:
{{
  "row_id": <row identifier exactly as given>,
  "corrections": [
    {{
      "field": <one of: {allowed_fields}>,
      "old_value": <original value>,
      "new_value": <corrected value, must satisfy that field's hard constraint>,
      "reason": <cite 2+ specific feature values that justify this correction where possible>,
      "confidence": <float 0-1, your confidence in THIS correction>
    }}
  ]
}}
Rows needing no correction still appear, with an empty "corrections" list.
Rules: preserve row order; do not add/drop rows; never omit or rename keys; never output
null/NaN for a value that exists; valid JSON only (RFC 8259)."""