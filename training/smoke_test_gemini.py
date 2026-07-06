"""
training/smoke_test_gemini.py

Manual smoke test that makes a real call to the configured LLM
provider. NOT a pytest test (it used to be named " test_gemini.py",
which matched pytest's `test_*.py` glob and would fire a live network
call as a side effect of test collection). Renamed out of discovery
and guarded behind `__main__`.

Run explicitly with (from the `training/` directory):
    python smoke_test_gemini.py
"""

import json

from config import Config
from llm_client import LLMClient


def main() -> None:
    cfg = Config()
    client = LLMClient(cfg)

    response = client.complete(
        system_prompt="""
You are a JSON API.
Respond ONLY with valid JSON.
""",
        user_prompt="""
Differentiate between Machine Learning (ML) and Deep Learning (DL).

Return JSON in this format:

{
    "machine_learning": "...",
    "deep_learning": "...",
    "key_differences": [
        "...",
        "...",
        "..."
    ]
}
""",
    )

    print(response.text)
    print(json.loads(response.text))


if __name__ == "__main__":
    main()