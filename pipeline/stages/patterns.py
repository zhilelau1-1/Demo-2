"""REPORTS_INTERPRETED -> PATTERNS_GROUPED: Stage 2, a separate LLM call over the Stage 1 interpretations
joined with part metadata, station, timestamp, confidence, measurements and the translated notes."""

from .. import prompts, vocab
from ..artifacts import write_json
from ..context import INTERPRETED_FILE, NORMALISED_FILE, PATTERNS_FILE, StageResult
from ..scoring import timestamp_key

STAGE = "pattern_grouping"


def build_payload(normalised, interpreted):
    interp = {i["report_id"]: i for i in interpreted}
    reports = []
    for r in normalised:
        i = interp[r["report_id"]]
        reports.append({
            "report_id": r["report_id"],
            "part_id": r["part_id"],
            "part_name": r["part_name"],
            "product_line": r["product_line"],
            "station": r["station"],
            "timestamp": r["timestamp"],
            "confidence": r["confidence"],
            "measurements": r["measurements"],
            "image_count": r["image_count"],
            "model_finding": r["model_finding"],
            "note_for_interpretation": r["note_for_interpretation"],
            "defect_category": i["defect_category"],
            "severity": i["severity"],
            "disposition": i["disposition"],
            "reason_short": i["reason_short"],
        })
    return {"reports": reports}


def validate_patterns(data, report_ids):
    errors = []
    seen_sets = set()
    for n, p in enumerate(data.get("patterns", []), start=1):
        ids = p.get("supporting_report_ids", [])
        unknown = [i for i in ids if i not in report_ids]
        if unknown:
            errors.append(f"pattern {n}: unknown report IDs {unknown}")
        if len(set(ids)) < 2:
            errors.append(f"pattern {n}: needs at least 2 distinct supporting reports, got {ids}")
        key = frozenset(ids)
        if key in seen_sets:
            errors.append(f"pattern {n}: duplicates another pattern's report set {sorted(ids)}")
        seen_sets.add(key)
        err = vocab.vocab_error("pattern_strength", p.get("pattern_strength"))
        if err:
            errors.append(f"pattern {n}: {err}")
        for field in ("title", "likely_driver", "recommended_action"):
            if not (p.get(field) or "").strip():
                errors.append(f"pattern {n}: {field} is empty")
    return errors


def run(ctx):
    normalised = ctx.load(NORMALISED_FILE)
    interpreted = ctx.load(INTERPRETED_FILE)
    report_ids = [r["report_id"] for r in normalised]
    instructions, schema = prompts.pattern_grouping(report_ids)
    data = ctx.llm.call(
        stage=STAGE,
        pipeline_state="PATTERNS_GROUPED",
        instructions=instructions,
        payload=build_payload(normalised, interpreted),
        schema=schema,
        validate=lambda d: validate_patterns(d, set(report_ids)),
        input_artifacts=[ctx.out(INTERPRETED_FILE), ctx.out(NORMALISED_FILE)],
        output_artifact=ctx.out(PATTERNS_FILE),
    )

    norm = {r["report_id"]: r for r in normalised}
    interp = {i["report_id"]: i for i in interpreted}
    order = lambda rid: (timestamp_key(norm[rid]["timestamp"]), rid)
    raw = []
    for p in data["patterns"]:
        ids = sorted(set(p["supporting_report_ids"]), key=order)
        raw.append((order(ids[0]), p["title"].strip(), ids, p))
    # Pattern IDs are assigned by code in a deterministic order (earliest supporting report first).
    raw.sort(key=lambda t: (t[0], t[1]))
    patterns = []
    for n, (_, title, ids, p) in enumerate(raw, start=1):
        patterns.append({
            "pattern_id": f"P{n:02d}",
            "title": title,
            "supporting_report_ids": ids,
            "pattern_strength": p["pattern_strength"],
            "likely_driver": p["likely_driver"].strip(),
            "recommended_action": p["recommended_action"].strip(),
            # Derived by code from the member reports:
            "defect_categories": sorted({interp[i]["defect_category"] for i in ids}),
            "part_ids": sorted({norm[i]["part_id"] for i in ids}),
            "stations": sorted({norm[i]["station"] for i in ids}),
            "first_seen": norm[ids[0]]["timestamp"],
            "last_seen": norm[ids[-1]]["timestamp"],
        })
    if len(patterns) < 2:
        ctx.state.warn(f"only {len(patterns)} recurring pattern(s) identified")
    write_json(ctx.out(PATTERNS_FILE), patterns)
    return StageResult(
        inputs=[ctx.out(INTERPRETED_FILE), ctx.out(NORMALISED_FILE)],
        outputs=[ctx.out(PATTERNS_FILE)],
        details={"pattern_count": len(patterns)},
    )
