"""REPORTS_NORMALISED -> REPORTS_INTERPRETED: Stage 1, one LLM call over all normalised reports."""

from .. import prompts, vocab
from ..artifacts import write_json
from ..context import INTERPRETED_FILE, NORMALISED_FILE, StageResult

STAGE = "report_interpretation"
PAYLOAD_FIELDS = (
    "report_id", "part_id", "part_name", "product_line", "station", "timestamp",
    "note_for_interpretation", "model_finding", "confidence", "measurements", "image_count",
)


def validate_interpretations(data, report_ids):
    errors = []
    items = data.get("interpretations", [])
    ids = [i.get("report_id") for i in items]
    if sorted(ids) != sorted(report_ids):
        errors.append(f"expected exactly one interpretation per report {sorted(report_ids)}, got {ids}")
    for i in items:
        for field in ("defect_category", "severity", "disposition"):
            err = vocab.vocab_error(field, i.get(field))
            if err:
                errors.append(f"{i.get('report_id')}: {err}")
        if not (i.get("reason_short") or "").strip():
            errors.append(f"{i.get('report_id')}: reason_short is empty")
    return errors


def run(ctx):
    normalised = ctx.load(NORMALISED_FILE)
    report_ids = [r["report_id"] for r in normalised]
    instructions, schema = prompts.interpretation(report_ids)
    data = ctx.llm.call(
        stage=STAGE,
        pipeline_state="REPORTS_INTERPRETED",
        instructions=instructions,
        payload={"reports": [{k: r[k] for k in PAYLOAD_FIELDS} for r in normalised]},
        schema=schema,
        validate=lambda d: validate_interpretations(d, report_ids),
        input_artifacts=[ctx.out(NORMALISED_FILE)],
        output_artifact=ctx.out(INTERPRETED_FILE),
    )
    by_id = {i["report_id"]: i for i in data["interpretations"]}
    records = []
    for r in normalised:
        i = by_id[r["report_id"]]
        records.append({
            "report_id": r["report_id"],
            "defect_category": i["defect_category"],
            "severity": i["severity"],
            "disposition": i["disposition"],
            "reason_short": i["reason_short"].strip(),
            # Carried from normalisation by code, never taken from the model.
            "translated": r["translated"],
            "original_language": r["original_language"],
        })
    write_json(ctx.out(INTERPRETED_FILE), records)
    counts = {}
    for rec in records:
        counts[rec["defect_category"]] = counts.get(rec["defect_category"], 0) + 1
    return StageResult(
        inputs=[ctx.out(NORMALISED_FILE)],
        outputs=[ctx.out(INTERPRETED_FILE)],
        details={"category_counts": counts},
    )
