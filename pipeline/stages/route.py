"""CASES_SELECTED -> ROUTING_COMPLETE: Stage 3, a separate LLM call routing flagged cases to owner teams."""

from .. import prompts, vocab
from ..artifacts import write_json
from ..context import FLAGGED_FILE, PATTERNS_FILE, ROUTING_FILE, SCORES_FILE, StageResult, flagged_cases

STAGE = "case_routing"


def validate_routes(data, flagged_ids):
    errors = []
    routes = data.get("routes", [])
    ids = [r.get("report_id") for r in routes]
    if sorted(ids) != sorted(flagged_ids):
        errors.append(f"expected exactly one route per flagged case {sorted(flagged_ids)}, got {ids}")
    for r in routes:
        teams = r.get("teams") or []
        if not teams:
            errors.append(f"{r.get('report_id')}: teams is empty")
        for t in teams:
            err = vocab.vocab_error("team", t)
            if err:
                errors.append(f"{r.get('report_id')}: {err}")
        if not (r.get("briefing_note") or "").strip():
            errors.append(f"{r.get('report_id')}: briefing_note is empty")
    return errors


def run(ctx):
    cases = flagged_cases(ctx)
    patterns = ctx.load(PATTERNS_FILE)
    flagged_ids = [c["report_id"] for c in cases]
    instructions, schema = prompts.routing(flagged_ids)
    data = ctx.llm.call(
        stage=STAGE,
        pipeline_state="ROUTING_COMPLETE",
        instructions=instructions,
        payload={"flagged_cases": cases, "patterns": patterns, "owner_teams": vocab.OWNER_TEAM_RESPONSIBILITIES},
        schema=schema,
        validate=lambda d: validate_routes(d, flagged_ids),
        input_artifacts=[ctx.out(FLAGGED_FILE), ctx.out(SCORES_FILE), ctx.out(PATTERNS_FILE)],
        output_artifact=ctx.out(ROUTING_FILE),
    )
    by_id = {r["report_id"]: r for r in data["routes"]}
    routing = []
    for c in cases:
        r = by_id[c["report_id"]]
        routing.append({
            "report_id": c["report_id"],
            "rank": c["rank"],
            "priority_score": c["priority_score"],
            "teams": list(dict.fromkeys(r["teams"])),
            "briefing_note": r["briefing_note"].strip(),
        })
    write_json(ctx.out(ROUTING_FILE), routing)
    return StageResult(
        inputs=[ctx.out(FLAGGED_FILE), ctx.out(SCORES_FILE), ctx.out(PATTERNS_FILE)],
        outputs=[ctx.out(ROUTING_FILE)],
        details={"routes": {r["report_id"]: r["teams"] for r in routing}},
    )
