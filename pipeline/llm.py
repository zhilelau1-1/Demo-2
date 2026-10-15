"""LLM access layer: provider selection, JSON-schema calls, validation retries, call logging and a
replay cache.

Modes:
  live    - call the Anthropic API (requires ANTHROPIC_API_KEY and the `anthropic` package)
  offline - deterministic rule-based provider (no network); logged as provider "offline-rules"
  replay  - serve recorded live responses from .llm_cache/ by prompt hash; fails on a cache miss
  auto    - live when ANTHROPIC_API_KEY is set, otherwise offline

Only an explicitly supplied ANTHROPIC_API_KEY enables live calls. Saved logins (`ant auth login`
profiles, OAuth tokens) are deliberately ignored, so a run never bills whoever happens to be logged in.
"""

import json
import os
import time
import uuid
from pathlib import Path

from . import offline
from .artifacts import canonical_json, display_path, read_json, sha256_text, utc_now, write_json
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "medium"
MAX_ATTEMPTS = 2


class LLMError(RuntimeError):
    pass


def anthropic_available():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model, effort):
        import anthropic

        self._anthropic = anthropic
        # Pass the key explicitly so the SDK never falls back to a saved login profile.
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.effort = effort
        self.use_fallbacks = os.environ.get("LLM_FALLBACKS", "default") != "off"

    def complete(self, stage, system, user, schema, payload):
        kwargs = {}
        if self.use_fallbacks:
            # Server-side refusal fallback: a declined request is re-run on Anthropic's recommended model.
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                **kwargs,
            )
        except self._anthropic.APIError as e:
            raise LLMError(f"Anthropic API error during {stage}: {e}") from e
        if response.stop_reason == "refusal":
            raise LLMError(f"model refused the {stage} request")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{stage} response hit max_tokens and was truncated")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError(f"{stage} response contained no text block")
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        return text, {"served_model": response.model, "usage": usage, "request_id": response._request_id}


class OfflineProvider:
    name = "offline-rules"
    model = offline.RULES_VERSION

    def complete(self, stage, system, user, schema, payload):
        data = offline.respond(stage, payload)
        return json.dumps(data, ensure_ascii=False), {"served_model": self.model}


class LLMClient:
    def __init__(self, mode, out_dir, run_id, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT, cache_dir=".llm_cache"):
        self.out_dir = Path(out_dir)
        self.run_id = run_id
        self.cache_dir = Path(cache_dir)
        self.log_path = self.out_dir / "llm_calls.jsonl"
        self.io_dir = self.out_dir / "llm_io"
        self.mode = mode
        if mode == "auto":
            mode = "live" if anthropic_available() else "offline"
        self.resolved_mode = mode
        if mode == "live":
            if not anthropic_available():
                raise LLMError("live mode needs the `anthropic` package and ANTHROPIC_API_KEY set")
            self.provider = AnthropicProvider(model, effort)
        elif mode == "offline":
            self.provider = OfflineProvider()
        elif mode == "replay":
            self.provider = None
        else:
            raise LLMError(f"unknown LLM mode {mode!r}")
        self.model = model if mode in ("live", "replay") else OfflineProvider.model
        self.effort = effort

    def describe(self):
        return {
            "requested_mode": self.mode,
            "mode": self.resolved_mode,
            "provider": self.provider.name if self.provider else "replay",
            "model": self.model,
            "effort": self.effort if self.resolved_mode != "offline" else None,
            "prompt_version": PROMPT_VERSION,
        }

    def reset_log(self, keep_states=None):
        """Start a fresh log, or (on resume) keep only records from stages that are not re-run."""
        if keep_states is None or not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")
            return
        kept = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("pipeline_state") in keep_states:
                kept.append(line)
        self.log_path.write_text("".join(l + "\n" for l in kept), encoding="utf-8")

    def _log(self, record):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _cache_path(self, prompt_hash):
        return self.cache_dir / f"{sha256_text(prompt_hash + '|' + self.model)}.json"

    def _complete(self, stage, system, user, schema, payload, prompt_hash):
        if self.resolved_mode == "replay":
            path = self._cache_path(prompt_hash)
            if not path.exists():
                raise LLMError(f"replay cache miss for {stage} (prompt_hash {prompt_hash[:12]}); run in live mode first")
            cached = read_json(path)
            return cached["response_text"], {"served_model": cached.get("served_model"), "replayed_from": cached["provider"]}
        text, meta = self.provider.complete(stage, system, user, schema, payload)
        if self.resolved_mode == "live":
            write_json(self._cache_path(prompt_hash), {
                "provider": self.provider.name, "model": self.model, "served_model": meta.get("served_model"),
                "stage": stage, "prompt_hash": prompt_hash, "created_at": utc_now(), "response_text": text,
            })
        return text, meta

    def call(self, *, stage, pipeline_state, instructions, payload, schema, validate,
             input_artifacts, output_artifact):
        """Run one logical LLM call and return validated JSON.

        `validate(data)` returns a list of error strings. Invalid output is retried once with the errors
        fed back; every attempt is logged, and exactly one attempt per stage can have status "success".
        """
        input_path = self.io_dir / f"{stage}.input.json"
        write_json(input_path, payload)
        feedback = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            user = f"{instructions}\n\nINPUT DATA (JSON):\n{json.dumps(payload, indent=2, ensure_ascii=False)}{feedback}"
            prompt_hash = sha256_text(canonical_json({"system": SYSTEM_PROMPT, "user": user, "schema": schema}))
            record = {
                "stage": stage,
                "pipeline_state": pipeline_state,
                "timestamp": utc_now(),
                "provider": self.provider.name if self.provider else "replay",
                "model": self.model,
                "prompt_hash": prompt_hash,
                "input_artifacts": [display_path(p) for p in input_artifacts] + [display_path(input_path)],
                "output_artifact": display_path(output_artifact),
                "run_id": self.run_id,
                "call_id": uuid.uuid4().hex[:12],
                "attempt": attempt,
                "mode": self.resolved_mode,
                "prompt_version": PROMPT_VERSION,
            }
            started = time.monotonic()
            try:
                text, meta = self._complete(stage, SYSTEM_PROMPT, user, schema, payload, prompt_hash)
            except LLMError as e:
                record.update(status="error", error=str(e), latency_ms=int((time.monotonic() - started) * 1000))
                self._log(record)
                raise
            record["latency_ms"] = int((time.monotonic() - started) * 1000)
            record["response_hash"] = sha256_text(text)
            record.update({k: v for k, v in meta.items() if v is not None})
            try:
                data = json.loads(text)
                errors = validate(data)
            except json.JSONDecodeError as e:
                data, errors = None, [f"response is not valid JSON: {e}"]
            if not errors:
                record["status"] = "success"
                self._log(record)
                write_json(self.io_dir / f"{stage}.response.json", data)
                return data
            record.update(status="invalid_output", validation_errors=errors[:20])
            self._log(record)
            feedback = (
                "\n\nYOUR PREVIOUS OUTPUT WAS REJECTED by validation with these errors:\n- "
                + "\n- ".join(errors[:20])
                + "\nReturn a corrected, complete output."
            )
        raise LLMError(f"{stage}: output failed validation after {MAX_ATTEMPTS} attempts: {errors[:5]}")
