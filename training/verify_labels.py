#!/usr/bin/env python3
"""
verify_labels.py
================
Main orchestrator for the LLM-based dataset label verification pipeline.

Contains (almost) no business logic — only orchestration:

    Load config
      -> Load CSV
      -> Infer schema
      -> Resume checkpoint
      -> Create batches
      -> For every batch:
           generate prompt -> call LLM -> validate -> retry/repair if needed
           -> save checkpoint
      -> Merge everything
      -> Export outputs
      -> Generate report

Usage:
    python verify_labels.py --input data/dataset.csv --provider openrouter
    python verify_labels.py --input data/dataset.csv --dry-run   # no API calls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from config import Config
from schema import infer_schema
from batcher import Batcher
from prompts import build_batch_prompt_bundle, build_repair_prompt_bundle
from llm_client import build_llm_client, LLMError
from validator import validate_response, check_decision_plausibility
import decision_plausibility as dp
from checkpoint import CheckpointManager, CheckpointState
from merger import apply_corrections, write_outputs
from utils import setup_logger, print_progress, safe_json_loads, checksum_of_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM dataset label verification pipeline")
    p.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    p.add_argument("--input", type=str, default=None, help="Input CSV path")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--provider", type=str, default=None,
                    choices=["deepseek", "openai", "gemini", "qwen", "claude", "openrouter"])
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--cost-limit", type=float, default=None, dest="cost_limit_usd")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint")
    p.add_argument("--clear-checkpoint", action="store_true", help="Delete checkpoint before running")
    p.add_argument("--dry-run", action="store_true", help="Use a mock LLM client (no network calls)")
    return p.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    overrides = {
        "input_csv": args.input,
        "output_dir": args.output_dir,
        "provider": args.provider,
        "model": args.model,
        "api_key": args.api_key,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "cost_limit_usd": args.cost_limit_usd,
        "resume_enabled": (not args.no_resume) if args.no_resume else None,
        "dry_run": args.dry_run or None,
        "clear_checkpoint": args.clear_checkpoint or None,
    }
    return Config.load(config_path=args.config, **overrides)


def run(config: Config) -> dict:
    logger = setup_logger("verify_labels", log_dir=config.log_dir)
    logger.info("Starting verification pipeline")
    logger.info(f"Provider={config.provider} model={config.model} dry_run={config.dry_run}")

    # ---- Load CSV -------------------------------------------------------
    input_path = Path(config.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {input_path}")

    # ---- Infer schema -----------------------------------------------------
    # Explicit Config overrides (id_column/decision_column/label_columns)
    # take precedence over the name-based heuristic when supplied -- see
    # schema.infer_schema()'s docstring.
    schema = infer_schema(
        df,
        id_column=config.id_column,
        decision_column=config.decision_column,
        label_columns=config.label_columns,
    )
    logger.info(
        f"Inferred schema: id={schema.id_column}, decision={schema.decision_column}, "
        f"scores={schema.score_columns}, labels={schema.label_columns}"
    )

    # ---- Build decision-plausibility context (once, over the full dataset) ---
    # See decision_plausibility.py: `decision` is a stochastic sample, not a
    # deterministic function of the other columns, so this is used only to
    # flag statistically implausible rows for escalation -- never to block
    # or fail the structural validation above.
    plausibility_ctx = None
    if schema.decision_column is not None:
        try:
            plausibility_ctx = dp.build_context(df, id_column=schema.id_column)
            logger.info(
                "Decision plausibility context built: reconstructed dist=%s",
                plausibility_ctx.diagnostics.get("reconstructed_mean_distribution"),
            )
        except ValueError as exc:
            logger.warning(f"Skipping decision-plausibility checks: {exc}")

    # ---- Resume checkpoint --------------------------------------------
    checkpoint_mgr = CheckpointManager(config.checkpoint_dir, input_csv_path=str(input_path))
    if config.clear_checkpoint:
        checkpoint_mgr.clear()

    # Computed once and reused for both the resume-staleness check and every
    # checkpoint save below, instead of re-hashing the whole input CSV file
    # on every single save() call.
    input_checksum = checksum_of_file(str(input_path))
    checkpoint_mgr.input_checksum = input_checksum
    if config.resume_enabled:
        state = checkpoint_mgr.resume(expect_input_checksum=input_checksum)
    else:
        state = CheckpointState()
    resume_from = state.last_completed_batch + 1
    if config.resume_from_batch is not None:
        # Explicit override: only allowed to skip forward, never to replay
        # batches the checkpoint already completed.
        resume_from = max(resume_from, config.resume_from_batch)
    if resume_from > 0:
        logger.info(f"Resuming from batch {resume_from} ({state.rows_completed} rows already done)")

    # ---- Create batches ----------------------------------------------
    batcher = Batcher(
        df, batch_size=config.batch_size, shuffle=config.shuffle,
        random_seed=config.random_seed, resume_from_batch=resume_from,
        max_batches=config.max_batches,
    )
    logger.info(f"Total batches: {batcher.total_batches}, remaining: {batcher.batches_remaining()}")

    # ---- LLM client -----------------------------------------------------
    llm_client = build_llm_client(config, logger=logger)

    all_llm_results = list(state.corrections or [])
    start_time = time.perf_counter()
    stopped_reason = None
     
    for batch in batcher:
        row_ids = batch.row_ids(schema.id_column)
        bundle = build_batch_prompt_bundle(batch.df, schema)

        validation = None
        response_text = ""
        last_user_prompt = bundle.user_prompt

        for attempt in range(1, config.max_retries + 1):
            try:
                llm_response = llm_client.complete(bundle.system_prompt, last_user_prompt)
            except LLMError as exc:
                logger.error(f"Batch {batch.index}: LLM call failed permanently: {exc}")
                stopped_reason = f"LLM call failed: {exc}"
                break

            response_text = llm_response.text
            state.total_api_calls += 1
            state.total_input_tokens += llm_response.input_tokens
            state.total_output_tokens += llm_response.output_tokens
            state.total_cost_usd += llm_response.cost_usd

            validation = validate_response(
                response_text, row_ids, schema,
                score_min=config.score_min, score_max=config.score_max,
                allowed_decisions=config.allowed_decisions,
            )
            if validation.is_valid:
                break

            logger.warning(
                f"Batch {batch.index} attempt {attempt}: validation failed "
                f"({len(validation.errors)} errors): {validation.errors[:3]}"
            )
            if attempt < config.max_retries:
                repair_bundle = build_repair_prompt_bundle(
                    schema, last_user_prompt, response_text, validation.errors,
                )
                last_user_prompt = repair_bundle.user_prompt

        if stopped_reason:
            break

        if validation is None or not validation.is_valid:
            logger.error(
                f"Batch {batch.index}: giving up after {config.max_retries} attempts. "
                "Skipping this batch's corrections; it will be retried on next resume "
                "if the checkpoint is not advanced."
            )
            # Do NOT advance the checkpoint past this batch, so a future run retries it.
            stopped_reason = f"Batch {batch.index} failed validation repeatedly"
            break

        all_llm_results.extend(validation.parsed)
        state.corrections = all_llm_results
        state.verified_row_ids.extend(row_ids)
        state.last_completed_batch = batch.index
        state.rows_completed += len(batch)

        if (batch.index + 1) % max(1, config.checkpoint_frequency) == 0:
            checkpoint_mgr.save(state)

        print_progress(state.rows_completed, batcher.total_rows, prefix="Verifying")

        if config.cost_limit_usd is not None and state.total_cost_usd >= config.cost_limit_usd:
            logger.warning(
                f"Cost limit reached (${state.total_cost_usd:.4f} >= "
                f"${config.cost_limit_usd:.4f}). Stopping early."
            )
            stopped_reason = "cost_limit_reached"
            break

    checkpoint_mgr.save(state)  # always persist final state, even on early stop

    # ---- Merge ------------------------------------------------------------
    merge_result = apply_corrections(df, schema, all_llm_results)
    output_paths = write_outputs(merge_result, config.output_dir)

    # ---- Decision plausibility escalations (non-blocking) -----------------
    # Runs against the FINAL (post-correction) decisions so a correction that
    # itself lands on an implausible decision also gets flagged.
    escalation_summary = {"checked": 0, "borderline": 0, "implausible": 0}
    if plausibility_ctx is not None:
        verified_df = merge_result.verified_df
        row_ids = (
            verified_df[schema.id_column].tolist()
            if schema.id_column and schema.id_column in verified_df.columns
            else verified_df.index.tolist()
        )
        report_p = check_decision_plausibility(
            df_full=df,
            row_ids=row_ids,
            decisions=verified_df[schema.decision_column].tolist(),
            schema=schema,
            ctx=plausibility_ctx,
        )
        escalation_summary["checked"] = report_p.checked
        escalation_summary["borderline"] = sum(
            1 for e in report_p.escalations if e["band"] == "borderline"
        )
        escalation_summary["implausible"] = sum(
            1 for e in report_p.escalations if e["band"] == "implausible"
        )
        if report_p.escalations:
            esc_path = Path(config.output_dir) / "escalations.csv"
            pd.DataFrame(report_p.escalations).to_csv(esc_path, index=False)
            output_paths["escalations"] = str(esc_path)
            logger.info(
                f"Wrote {len(report_p.escalations)} decision-plausibility "
                f"escalations ({escalation_summary['borderline']} borderline, "
                f"{escalation_summary['implausible']} implausible) to {esc_path}"
            )

    # ---- Report -------------------------------------------------------
    runtime_s = time.perf_counter() - start_time
    report = {
        "rows_total": len(df),
        "rows_verified": len(set(state.verified_row_ids)),
        "rows_corrected": merge_result.rows_corrected,
        "fields_corrected": merge_result.fields_corrected,
        "fields_attempted": merge_result.fields_attempted,
        "runtime_s": round(runtime_s, 2),
        "api_calls": state.total_api_calls,
        "input_tokens": state.total_input_tokens,
        "output_tokens": state.total_output_tokens,
        "cost_usd": round(state.total_cost_usd, 4),
        "provider": config.provider,
        "model": config.model,
        "stopped_reason": stopped_reason,
        "outputs": output_paths,
        "decision_plausibility": escalation_summary,
    }
    report_path = Path(config.output_dir) / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Report written to {report_path}")
    logger.info(json.dumps(report, indent=2, default=str))

    fully_complete = stopped_reason is None and state.rows_completed >= len(df)
    if fully_complete:
        checkpoint_mgr.clear()  # successful full run: no need to keep checkpoint
        logger.info("Run completed successfully; checkpoint cleared.")
    elif stopped_reason is None:
        logger.info(
            f"Run stopped after max_batches limit ({state.rows_completed}/{len(df)} rows "
            "done so far). Checkpoint kept so the run can resume."
        )

    return report


def main():
    args = parse_args()
    config = build_config_from_args(args)
    try:
        report = run(config)
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()