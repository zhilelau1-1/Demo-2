"""INPUTS_LOADED -> REPORTS_NORMALISED: join reports with the part catalog, normalise notes, and detect
and translate non-English notes (one batched LLM call). The original note is copied verbatim by code."""

import re

from .. import prompts
from ..artifacts import write_json
from ..context import CATALOG_FILE, NORMALISED_FILE, REPORTS_FILE, StageResult
from .load import load_inputs

STAGE = "normalisation_translation"
_LANG_CODE = re.compile(r"^[a-z]{2,3}$")


def clean_note(note):
    return " ".join(note.split())


def validate_translation(data, notes_by_id):
    errors = []
    items = data.get("notes", [])
    ids = [n.get("report_id") for n in items]
    if sorted(ids) != sorted(notes_by_id):
        errors.append(f"expected exactly one entry per report {sorted(notes_by_id)}, got {ids}")
    for n in items:
        rid, lang = n.get("report_id"), n.get("original_language", "")
        if not _LANG_CODE.match(lang or ""):
            errors.append(f"{rid}: original_language {lang!r} is not an ISO 639-1 code")
        if n.get("is_english") != (lang == "en"):
            errors.append(f"{rid}: is_english={n.get('is_english')} conflicts with original_language={lang!r}")
        if not n.get("is_english") and n.get("english_text") and clean_note(n["english_text"]) == notes_by_id.get(rid):
            errors.append(f"{rid}: note marked non-English but english_text is identical to the original")
    return errors


def run(ctx):
    reports, catalog, _ = load_inputs(ctx.input_dir)
    parts = {p["part_id"]: p for p in catalog}
    notes_by_id = {r["report_id"]: clean_note(r["operator_note"]) for r in reports}

    instructions, schema = prompts.translation(list(notes_by_id))
    data = ctx.llm.call(
        stage=STAGE,
        pipeline_state="REPORTS_NORMALISED",
        instructions=instructions,
        payload={"notes": [{"report_id": rid, "operator_note": note} for rid, note in notes_by_id.items()]},
        schema=schema,
        validate=lambda d: validate_translation(d, notes_by_id),
        input_artifacts=[ctx.inp(REPORTS_FILE)],
        output_artifact=ctx.out(NORMALISED_FILE),
    )
    detected = {n["report_id"]: n for n in data["notes"]}

    records, untranslated = [], []
    for r in reports:
        rid = r["report_id"]
        part = parts.get(r["part_id"])
        lang_info = detected[rid]
        if lang_info["is_english"]:
            note_for_interpretation, translated, status = notes_by_id[rid], False, "not_required"
        elif lang_info["english_text"].strip():
            note_for_interpretation, translated, status = clean_note(lang_info["english_text"]), True, "translated"
        else:
            note_for_interpretation, translated, status = notes_by_id[rid], False, "untranslated"
            untranslated.append(rid)
            ctx.state.warn(f"{rid}: note in {lang_info['original_language']} could not be translated by the {ctx.llm.describe()['provider']} provider")
        records.append({
            "report_id": rid,
            "part_id": r["part_id"],
            "part_name": part["part_name"] if part else "UNKNOWN",
            "product_line": part["product_line"] if part else "UNKNOWN",
            "catalog_match": part is not None,
            "station": r["station"],
            "timestamp": r["timestamp"],
            "model_finding": r["model_finding"],
            "confidence": r["confidence"],
            "measurements": r["measurements"],
            "image_count": r["image_count"],
            "original_operator_note": r["operator_note"],
            "note_for_interpretation": note_for_interpretation,
            "original_language": lang_info["original_language"],
            "translated": translated,
            "translation_status": status,
        })
    write_json(ctx.out(NORMALISED_FILE), records)
    return StageResult(
        inputs=[ctx.inp(REPORTS_FILE), ctx.inp(CATALOG_FILE)],
        outputs=[ctx.out(NORMALISED_FILE)],
        details={
            "translated_report_ids": [r["report_id"] for r in records if r["translated"]],
            "untranslated_report_ids": untranslated,
            "unmatched_parts": sorted({r["part_id"] for r in records if not r["catalog_match"]}),
        },
    )
