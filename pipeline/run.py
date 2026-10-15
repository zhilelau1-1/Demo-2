"""Pipeline orchestrator.

    python -m pipeline.run                         # full run (LLM mode auto: live if credentials, else offline)
    python -m pipeline.run --llm-mode offline      # deterministic rule-based provider, no network
    python -m pipeline.run --llm-mode replay       # re-run using recorded live LLM responses
    python -m pipeline.run --from-stage PRIORITIES_COMPUTED   # resume from a stage of the previous run
"""

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .context import CATALOG_FILE, MANIFEST_FILE, REPORTS_FILE, Context
from .llm import DEFAULT_EFFORT, DEFAULT_MODEL, LLMClient, LLMError
from .stages import explain, finish, interpret, load, normalise, patterns, route, score, selection
from .state import STAGE_ORDER, PipelineState, PipelineStateError, Stage, stage_index

STAGES = [
    (Stage.INPUTS_LOADED, load.run),
    (Stage.REPORTS_NORMALISED, normalise.run),
    (Stage.REPORTS_INTERPRETED, interpret.run),
    (Stage.PATTERNS_GROUPED, patterns.run),
    (Stage.PRIORITIES_COMPUTED, score.run),
    (Stage.CASES_SELECTED, selection.run),
    (Stage.ROUTING_COMPLETE, route.run),
    (Stage.EXPLANATIONS_GENERATED, explain.run),
    (Stage.VALIDATION_COMPLETE, finish.validate),
    (Stage.RESULTS_FINALISED, finish.finalise),
]
TOP_N = 5


def parse_args(argv):
    p = argparse.ArgumentParser(description="Inspection triage pipeline")
    p.add_argument("--input-dir", default=".", help="directory containing inspection_reports.json and parts_catalog.json")
    p.add_argument("--out-dir", default=".", help="directory for generated artifacts")
    p.add_argument("--llm-mode", choices=("auto", "live", "offline", "replay"), default="auto")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default=DEFAULT_EFFORT, choices=("low", "medium", "high", "xhigh", "max"))
    p.add_argument("--cache-dir", default=".llm_cache", help="where live LLM responses are recorded for replay")
    p.add_argument("--from-stage", choices=[s.value for s in STAGE_ORDER[1:]], help="resume the previous run from this stage")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [input_dir / REPORTS_FILE, input_dir / CATALOG_FILE]
    for p in input_paths:
        if not p.exists():
            print(f"error: input file {p} not found", file=sys.stderr)
            return 2

    try:
        if args.from_stage:
            state = PipelineState.resume(out_dir / MANIFEST_FILE, args.from_stage, input_paths)
            run_id = state.run_id
        else:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
            state = None
        llm = LLMClient(args.llm_mode, out_dir, run_id, model=args.model, effort=args.effort, cache_dir=args.cache_dir)
        if state is None:
            llm.reset_log()
            state = PipelineState(out_dir / MANIFEST_FILE, run_id, llm.describe(), input_paths)
        else:
            llm.reset_log(keep_states={s.value for s in STAGE_ORDER[: stage_index(args.from_stage)]})
            state.manifest["llm"] = llm.describe()
            state.save()
    except (PipelineStateError, LLMError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    info = llm.describe()
    print(f"run {run_id} · LLM {info['provider']} ({info['model']}, mode {info['mode']})")
    ctx = Context(input_dir=input_dir, out_dir=out_dir, llm=llm, state=state, top_n=TOP_N)
    for target, fn in STAGES:
        if stage_index(target) <= stage_index(state.current):
            continue
        try:
            state.require_next(target)
            result = fn(ctx)
            state.advance(target, result.inputs, result.outputs, result.details)
        except Exception as e:  # record the failure in the manifest, then report it
            state.finish("failed", error=f"{target.value}: {type(e).__name__}: {e}")
            print(f"[{target.value}] FAILED: {e}", file=sys.stderr)
            return 1
        print(f"[{target.value}] done")
    state.finish("complete")
    print(f"pipeline complete; artifacts in {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
