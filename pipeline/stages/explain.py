"""ROUTING_COMPLETE -> EXPLANATIONS_GENERATED: reviewer summaries for the flagged cases, plus the optional
shift-level analyses.

Code renders every fact (category, severity, disposition, score breakdown, pattern membership) and the
human review gate note; the LLM only writes the "why prioritised" narrative and the verification focus.
"""

from .. import prompts
from ..artifacts import utc_now, write_text
from ..context import FLAGGED_FILE, PATTERNS_FILE, ROUTING_FILE, SCORES_FILE, SUMMARIES_FILE, StageResult, flagged_cases
from . import analyses

STAGE = "explanation_generation"

GATE_CHECKS = {
    "solder_voiding": "confirm the void ratio on a fresh X-ray image of the affected joint(s) against the applicable acceptance criteria, and check whether other units from the same station and lot show the same signature",
    "pin_misalignment": "measure pin position with a calibrated method (not the automated estimate alone) against the drawing tolerance, and check the placement/fixture setup",
    "surface_scratch": "examine the mark under controlled lighting to decide whether it is cosmetic or breaks coating/plating, against the cosmetic acceptance standard, and identify the handling step that caused it",
    "no_material_issue": "confirm on the original images that no material defect is present before releasing the unit",
    "uncertain": "have a second inspector classify the defect from the unit and the original images, because the automated interpretation is not conclusive",
}


def validate_explanations(data, flagged_ids):
    errors = []
    items = data.get("explanations", [])
    ids = [e.get("report_id") for e in items]
    if sorted(ids) != sorted(flagged_ids):
        errors.append(f"expected exactly one explanation per flagged case {sorted(flagged_ids)}, got {ids}")
    for e in items:
        for field in ("why_prioritised", "verification_focus"):
            if not (e.get(field) or "").strip():
                errors.append(f"{e.get('report_id')}: {field} is empty")
    return errors


def gate_note(case, teams, verification_focus):
    signoff = ", ".join(dict.fromkeys(["Quality", *teams]))
    note = (
        f"Before any production decision (release, rework, scrap or lot disposition), a qualified reviewer must "
        f"{GATE_CHECKS[case['defect_category']]}. Specific check: {verification_focus.strip()}"
    )
    if case["translated"]:
        note += f" The operator note was translated from '{case['original_language']}'; confirm the translation with a fluent reviewer or the operator."
    note += f" The AI interpretation and priority score are advisory only; sign-off required from: {signoff}."
    return note


def _score_line(case):
    c = case["score_components"]
    measurement = f"measurement {c['measurement_bonus']}"
    if case["measurement_hits"]:
        measurement += " (" + ", ".join(f"{h['measurement']} {h['value']} >= {h['threshold']}" for h in case["measurement_hits"]) + ")"
    pattern = f"pattern membership {c['pattern_membership_bonus']}"
    if case["pattern_ids"]:
        pattern += f" ({', '.join(case['pattern_ids'])})"
    return (
        f"base {c['base_priority']} ({case['severity']}) + confidence {c['confidence_bonus']} + "
        f"evidence {c['evidence_bonus']} ({case['image_count']} images) + {measurement} + {pattern} + "
        f"disposition {c['disposition_bonus']} ({case['disposition']}) = **{case['priority_score']}**"
    )


def render_summaries(cases, patterns, routing, explanations, total_reports, llm_info, run_id):
    patterns_by_id = {p["pattern_id"]: p for p in patterns}
    lines = [
        f"# Reviewer Summaries: Top {len(cases)} Flagged Cases",
        "",
        f"Run `{run_id}` · generated {utc_now()} · interpretation by {llm_info['provider']} ({llm_info['model']}).",
        "",
        "Priority scores are computed deterministically in code from the interpreted severity and disposition, "
        "confidence, image evidence, measurement thresholds and recurring-pattern membership. "
        "The AI outputs are decision support only; every case below requires the human review gate before any production decision.",
        "",
    ]
    for case in cases:
        rid = case["report_id"]
        route = routing[rid]
        expl = explanations[rid]
        if case["pattern_ids"]:
            membership = "Yes: " + "; ".join(
                f"{pid} \"{patterns_by_id[pid]['title']}\" ({patterns_by_id[pid]['pattern_strength']})" for pid in case["pattern_ids"]
            )
        else:
            membership = "No (isolated case)"
        lines += [
            f"## {case['rank']}. {rid}: {case['part_name']} ({case['part_id']})",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Report ID | {rid} |",
            f"| Part | {case['part_name']} ({case['part_id']}, {case['product_line']}) |",
            f"| Station / time | {case['station']} · {case['timestamp']} |",
            f"| Defect category | {case['defect_category']} |",
            f"| Severity | {case['severity']} |",
            f"| Disposition | {case['disposition']} |",
            f"| Priority score | {case['priority_score']} (rank {case['rank']} of {total_reports}) |",
            f"| Recurring pattern | {membership} |",
            f"| Routed to | {', '.join(route['teams'])} |",
            "",
            f"**Score breakdown:** {_score_line(case)}",
            "",
            f"**Interpretation:** {case['reason_short']}",
            "",
        ]
        if case["translated"]:
            lines += [f"**Operator note (translated from {case['original_language']}):** {case['note_for_interpretation']}", ""]
        lines += [
            f"**Why it was prioritised:** {expl['why_prioritised'].strip()}",
            "",
            f"**Routing brief:** {route['briefing_note']}",
            "",
            f"**Human review gate:** {gate_note(case, route['teams'], expl['verification_focus'])}",
            "",
        ]
    return "\n".join(lines)


def run(ctx):
    cases = flagged_cases(ctx)
    patterns = ctx.load(PATTERNS_FILE)
    routing = {r["report_id"]: r for r in ctx.load(ROUTING_FILE)}
    flagged_ids = [c["report_id"] for c in cases]
    instructions, schema = prompts.explanation(flagged_ids)
    data = ctx.llm.call(
        stage=STAGE,
        pipeline_state="EXPLANATIONS_GENERATED",
        instructions=instructions,
        payload={
            "flagged_cases": [{**c, "teams": routing[c["report_id"]]["teams"]} for c in cases],
            "patterns": patterns,
        },
        schema=schema,
        validate=lambda d: validate_explanations(d, flagged_ids),
        input_artifacts=[ctx.out(FLAGGED_FILE), ctx.out(SCORES_FILE), ctx.out(PATTERNS_FILE), ctx.out(ROUTING_FILE)],
        output_artifact=ctx.out(SUMMARIES_FILE),
    )
    explanations = {e["report_id"]: e for e in data["explanations"]}
    total = len(ctx.load(SCORES_FILE)["scores"])
    text = render_summaries(cases, patterns, routing, explanations, total, ctx.llm.describe(), ctx.state.run_id)
    write_text(ctx.out(SUMMARIES_FILE), text)

    analysis_outputs, analysis_stages = analyses.run_all(ctx)
    return StageResult(
        inputs=[ctx.out(FLAGGED_FILE), ctx.out(SCORES_FILE), ctx.out(PATTERNS_FILE), ctx.out(ROUTING_FILE)],
        outputs=[ctx.out(SUMMARIES_FILE), *analysis_outputs],
        details={"llm_stages": [STAGE, *analysis_stages], "summarised_report_ids": flagged_ids},
    )
