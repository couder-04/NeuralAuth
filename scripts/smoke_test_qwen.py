"""
scripts/smoke_test_qwen.py

Manual smoke test for the Qwen intent-extraction model. Downloads and
loads the model, then runs one generation -- useful for verifying a
new environment/GPU can actually serve the model, but NOT a pytest
test (hence living under scripts/, not tests/, and guarded by
`__main__` so importing this module never has side effects).

Run explicitly with:
    python scripts/smoke_test_qwen.py
"""

from transformers import pipeline


def main() -> None:
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
    )

    messages = [
        {
            "role": "user",
            "content": "Say hello",
        }
    ]

    result = pipe(
        messages,
        max_new_tokens=20,
        return_full_text=False,
        do_sample=False,
    )
    print(result)


if __name__ == "__main__":
    main()