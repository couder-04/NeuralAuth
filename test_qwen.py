from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
)

messages = [
    {
        "role": "user",
        "content": "Say hello"
    }
]

print(
    pipe(
        messages,
        max_new_tokens=20,
        return_full_text=False,
        do_sample=False,
    )
)