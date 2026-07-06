"""
llm_client.py
=============
Single interface to the LLM provider (DeepSeek by default; also supports
OpenAI, Gemini, Qwen, and Claude via configuration).

Public surface:
    LLMClient.complete(system_prompt, user_prompt) -> LLMResponse

Internally handles retries, timeout, exponential backoff, rate limiting,
API errors, malformed responses, token usage, and cost estimation.

This file NEVER touches the dataset or infers schema.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from config import Config, PROVIDER_DEFAULTS
from utils import estimate_tokens, estimate_cost, new_uuid, pretty


def _clean_json(text: str) -> str:
    """Strips markdown fences and introductory conversational text 
    to extract a pure JSON object or array string.
    """
    text = text.strip()

    if "```" in text:
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    # Find the starting position of the actual JSON payload
    start = min(
        [i for i in [text.find("{"), text.find("[")] if i != -1],
        default=-1
    )

    if start != -1:
        text = text[start:]

    return text.strip()


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


@dataclass
class LLMResponse:
    text: str                     # raw text content returned by the model
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    provider: str
    model: str
    request_id: str
    attempts: int = 1


class _RateLimiter:
    """Simple token-bucket-ish limiter based on requests-per-minute."""

    def __init__(self, requests_per_minute: int):
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._last_call = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        elapsed = time.time() - self._last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.time()


class LLMClient:
    """Provider-agnostic chat-completion client."""

    def __init__(self, config: Config, logger=None):
        self.config = config
        self.logger = logger
        self.provider = config.provider
        self.model = config.model
        self.api_key = config.api_key
        self.base_url = config.base_url
        self._rate_limiter = _RateLimiter(config.requests_per_minute)
        self.total_cost = 0.0  # Track cumulative cost

        if config.log_dir:
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        if not self.api_key and self.provider in PROVIDER_DEFAULTS:
            env_name = PROVIDER_DEFAULTS[self.provider]["api_key_env"]
            self._warn(
                f"No API key configured for provider '{self.provider}'. "
                f"Set the {env_name} environment variable, or use dry_run=True."
            )

    # ------------------------------------------------------------------
    def _warn(self, msg: str):
        if self.logger:
            self.logger.warning(msg)
        else:
            print(f"[WARN] {msg}")

    def _log_call(self, request_id: str, system_prompt: str, user_prompt: str,
                   response_text: str, meta: dict):
        if not self.config.log_dir:
            return
        record = {
            "request_id": request_id,
            "provider": self.provider,
            "model": self.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_text": response_text,
            "meta": meta,
        }
        path = Path(self.config.log_dir) / f"{request_id}.json"
        path.write_text(pretty(record))

    # ------------------------------------------------------------------
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Send one chat-completion request, with retries + backoff."""
        
        # Look-ahead cost limit verification before execution
        if self.config.cost_limit_usd is not None:
            estimated_request_cost = estimate_cost(
                self.model,
                estimate_tokens(system_prompt) + estimate_tokens(user_prompt),
                self.config.max_tokens if self.config.max_tokens is not None else 0
            )
            if self.total_cost + estimated_request_cost > self.config.cost_limit_usd:
                raise LLMError("Cost limit would be exceeded.")

        request_id = new_uuid()
        attempts = 0
        last_error: Optional[Exception] = None

        while attempts < self.config.max_retries:
            attempts += 1
            self._rate_limiter.wait()
            start = time.perf_counter()
            try:
                result = self._dispatch(system_prompt, user_prompt)
                text = result["content"]
                usage = result.get("usage", {})
                latency = time.perf_counter() - start

                # Accurately fallback to estimates if exact counts aren't provided
                in_tok = usage.get(
                    "prompt_tokens", 
                    estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
                )
                out_tok = usage.get(
                    "completion_tokens", 
                    estimate_tokens(text)
                )
                
                cost = estimate_cost(self.model, in_tok, out_tok)
                self.total_cost += cost  # Accumulate session cost

                self._log_call(request_id, system_prompt, user_prompt, text,
                                {"latency_s": latency, "attempts": attempts,
                                 "input_tokens": in_tok, "output_tokens": out_tok,
                                 "cost_usd": cost, "total_cost_usd": self.total_cost})

                return LLMResponse(
                    text=text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=cost,
                    latency_s=latency,
                    provider=self.provider,
                    model=self.model,
                    request_id=request_id,
                    attempts=attempts,
                )
            except Exception as exc:  # noqa: BLE001 - broad on purpose, we classify below
                last_error = exc
                if attempts >= self.config.max_retries:
                    break
                backoff = min(
                    self.config.retry_backoff_max,
                    self.config.retry_backoff_base * (2 ** (attempts - 1)),
                )
                backoff += random.uniform(0, 0.5)  # jitter
                self._warn(
                    f"[{request_id}] attempt {attempts} failed: {exc}. "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)

        raise LLMError(
            f"LLM call failed after {attempts} attempts (request_id={request_id}): {last_error}"
        )

    # ------------------------------------------------------------------
    def _dispatch(self, system_prompt: str, user_prompt: str) -> dict:
        """Route to the correct provider-specific HTTP call. Returns dict with content/usage."""
        if not self.api_key:
            raise LLMError(
                f"No API key set for provider '{self.provider}'. "
                "Configure it or use dry_run mode."
            )

        if self.provider in ("openrouter"):
            return self._call_openai_compatible(system_prompt, user_prompt)
        elif self.provider == "claude":
            return self._call_anthropic(system_prompt, user_prompt)
        elif self.provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt)
        else:
            raise LLMError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------
    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> dict:
        provider_cfg = PROVIDER_DEFAULTS[self.provider]
        url = self.base_url.rstrip("/") + provider_cfg["chat_path"]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }

        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = "Transaction Label Verifier"
            headers["Accept"] = "application/json"
            body["provider"] = {
                "allow_fallbacks": False
            }

        resp = requests.post(url, headers=headers, json=body,
                              timeout=self.config.request_timeout)

        try:
            data = resp.json()
        except Exception:
            data = {}

        if resp.status_code >= 400:
            message = data.get("error", {}).get("message") or resp.text
            raise LLMError(f"{resp.status_code}: {message}")

        try:
            raw_content = data["choices"][0]["message"]["content"]
            return {
                "content": _clean_json(raw_content),
                "usage": data.get("usage", {}),
            }
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Malformed response shape: {data}") from exc

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> dict:
        url = self.base_url.rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        resp = requests.post(url, headers=headers, json=body,
                              timeout=self.config.request_timeout)
        
        try:
            data = resp.json()
        except Exception:
            data = {}

        if resp.status_code >= 400:
            message = data.get("error", {}).get("message") or resp.text
            raise LLMError(f"{resp.status_code}: {message}")

        try:
            parts = [b["text"] for b in data["content"] if b.get("type") == "text"]
            raw_content = "\n".join(parts)
            return {
                "content": _clean_json(raw_content),
                "usage": data.get("usage", {}),
            }
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Malformed response shape: {data}") from exc

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> dict:
        path = f"/models/{self.model}:generateContent"
        url = self.base_url.rstrip("/") + path
        params = {"key": self.api_key}
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
                "responseMimeType": "application/json",
            },
        }
        resp = requests.post(url, params=params, json=body,
                              timeout=self.config.request_timeout)
        
        try:
            data = resp.json()
        except Exception:
            data = {}

        if resp.status_code >= 400:
            message = data.get("error", {}).get("message") or resp.text
            raise LLMError(f"{resp.status_code}: {message}")

        try:
            parts = data["candidates"][0]["content"]["parts"]
            raw_content = "".join(p.get("text", "") for p in parts)
            
            # Translate Gemini's native usage response payload keys to unified OpenAI names
            usage_meta = data.get("usageMetadata", {})
            standard_usage = {
                "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                "completion_tokens": usage_meta.get("candidatesTokenCount", 0)
            }

            return {
                "content": _clean_json(raw_content),
                "usage": standard_usage,
            }
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Malformed response shape: {data}") from exc


class MockLLMClient:
    """Drop-in replacement for LLMClient used in --dry-run / tests.

    Returns syntactically valid 'no corrections needed' responses for every
    row it's given, without making any network calls. This lets the rest of
    the pipeline (batching, validation, checkpointing, merging) be exercised
    end-to-end offline.
    """

    def __init__(self, config: Config, logger=None):
        self.config = config
        self.logger = logger
        self.provider = "mock"
        self.model = "mock-model"

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        # Extract the row ids that were embedded in the user prompt so the
        # mock response has correct shape and passes validation.
        row_ids = []
        try:
            start = user_prompt.index("ROWS:\n") + len("ROWS:\n")
            payload = user_prompt[start:]
            rows = json.loads(payload.split("\n\nRespond")[0])
            row_ids = [r.get("_row_id") for r in rows]
        except Exception:
            row_ids = []

        results = [{"row_id": rid, "corrections": []} for rid in row_ids]
        text = json.dumps(results)

        return LLMResponse(
            text=text,
            input_tokens=estimate_tokens(system_prompt) + estimate_tokens(user_prompt),
            output_tokens=estimate_tokens(text),
            cost_usd=0.0,
            latency_s=0.001,
            provider=self.provider,
            model=self.model,
            request_id=new_uuid(),
            attempts=1,
        )


def build_llm_client(config: Config, logger=None):
    """Factory: returns a MockLLMClient in dry_run mode, else a real LLMClient."""
    if config.dry_run:
        return MockLLMClient(config, logger=logger)
    return LLMClient(config, logger=logger)