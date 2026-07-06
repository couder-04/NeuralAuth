# Dataset Label Verification Pipeline

A small, production-grade pipeline that uses an LLM (DeepSeek by default)
to verify and correct labels/scores/decisions in a CSV dataset, one batch
of rows at a time — with retries, validation, resumable checkpoints, and
a final report.

## Layout

```
training/
├── verify_labels.py     # Main orchestrator (CLI entry point)
├── prompts.py           # Prompt engineering (system/batch/repair prompts)
├── llm_client.py        # Provider-agnostic LLM wrapper + MockLLMClient
├── batcher.py            # Splits the dataset into batches
├── merger.py             # Applies corrections, writes final CSVs
├── checkpoint.py         # Save/load/resume run state
├── validator.py          # Validates every LLM response before it's applied
├── schema.py              # Infers dataset schema (id/decision/score columns)
├── config.py               # Central configuration (dataclass)
├── utils.py                # Logging, timing, hashing, token/cost estimates
├── requirements.txt
├── data/
│   └── dataset.csv       # Put your input CSV here
└── outputs/
    ├── verified_dataset.csv
    ├── corrections.csv
    ├── verification_report.json
    ├── llm_logs/           # One JSON file per real LLM request/response
    └── checkpoints/        # checkpoint.json (deleted on successful full run)
```

## Install

```bash
pip install -r requirements.txt
```

## Configure

Set an API key as an environment variable for whichever provider you use:

| Provider | Env var             | Default model       |
|----------|---------------------|----------------------|
| deepseek | `DEEPSEEK_API_KEY`  | deepseek-chat        |
| openai   | `OPENAI_API_KEY`    | gpt-4o-mini          |
| gemini   | `GEMINI_API_KEY`    | gemini-1.5-flash     |
| qwen     | `DASHSCOPE_API_KEY` | qwen-plus            |
| claude   | `ANTHROPIC_API_KEY` | claude-sonnet-4-6    |

Or pass `--api-key` directly (not recommended for shared shells/history).

You can also write a JSON config file and pass `--config path.json`; any
field from `config.py`'s `Config` dataclass can be set there. CLI flags
override the config file.

## Run

```bash
# Try it with no network calls at all (uses a mock client):
python verify_labels.py --input data/dataset.csv --dry-run

# Real run against DeepSeek:
export DEEPSEEK_API_KEY=sk-...
python verify_labels.py --input data/dataset.csv --provider deepseek --batch-size 64

# Resume is automatic: if a run is interrupted, just re-run the same
# command and it picks up from the last completed batch.
python verify_labels.py --input data/dataset.csv --no-resume   # ignore existing checkpoint
python verify_labels.py --input data/dataset.csv --clear-checkpoint  # wipe it and start over

# Cap spend / scope for testing:
python verify_labels.py --input data/dataset.csv --max-batches 2 --cost-limit 1.50
```

## How it works

1. **Schema inference** (`schema.py`) looks at column names (id, score,
   trust/risk, decision, reason, evidence-like text) so the pipeline
   doesn't need to hardcode your dataset's column names.
2. **Batching** (`batcher.py`) splits the dataframe into fixed-size
   batches, honoring `--max-batches` and any resume point.
3. For each batch, **`prompts.py`** builds a system + user prompt asking
   the model to return a JSON array of `{row_id, corrections: [...]}`.
4. **`llm_client.py`** calls the provider's chat-completions endpoint
   (or the Anthropic Messages API / Gemini generateContent shape, as
   appropriate) with retries, exponential backoff + jitter, and a simple
   requests-per-minute rate limiter. Every real request/response pair is
   logged to `outputs/llm_logs/`.
5. **`validator.py`** checks the JSON is well-formed, has the right
   fields, no duplicate/missing/extra row ids, matching batch size and
   order, and that scores/decisions are within allowed ranges. If it
   fails, `prompts.build_repair_prompt` asks the model to fix its own
   answer; this repeats up to `max_retries` times.
6. **`checkpoint.py`** saves progress after each batch (rows completed,
   token/cost totals, accumulated corrections, and a checksum of the
   input file so a changed input invalidates a stale checkpoint).
7. Once all batches are done (or a limit/error stops the run early),
   **`merger.py`** applies every correction to a copy of the original
   dataframe and writes `verified_dataset.csv` plus a `corrections.csv`
   audit trail (including any corrections that couldn't be applied, and
   why).
8. `verify_labels.py` writes a final `verification_report.json` with
   row/cost/token/runtime totals.

## Swapping LLM providers

Everything provider-specific lives in `llm_client.py`. To point at a new
model, set `--provider` (and optionally `--model`, `--base-url` via a
config file) — no other file needs to change.

## Testing without an API key

Pass `--dry-run` to use `MockLLMClient`, which returns syntactically
valid "no corrections needed" responses for every row so you can
exercise batching, validation, checkpointing, and merging end-to-end
offline.
