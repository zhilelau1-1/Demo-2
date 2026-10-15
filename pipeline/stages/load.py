"""INIT -> INPUTS_LOADED: read both input files from disk and check their schema."""

from ..artifacts import read_json
from ..context import CATALOG_FILE, REPORTS_FILE, StageResult

REPORT_FIELDS = ("report_id", "part_id", "station", "timestamp", "operator_note", "model_finding", "confidence", "measurements", "image_count")
CATALOG_FIELDS = ("part_id", "part_name", "product_line")
MEASUREMENTS = ("void_ratio", "misalignment_mm", "scratch_length_mm")


class InputError(ValueError):
    pass


def load_inputs(input_dir):
    """Return (reports, catalog, warnings) with defensive defaults applied. Raises InputError on bad input."""
    warnings = []
    try:
        reports = read_json(input_dir / REPORTS_FILE)
        catalog = read_json(input_dir / CATALOG_FILE)
    except FileNotFoundError as e:
        raise InputError(f"missing input file: {e.filename}") from e
    if not isinstance(reports, list) or not reports:
        raise InputError(f"{REPORTS_FILE} must be a non-empty JSON array")
    if not isinstance(catalog, list):
        raise InputError(f"{CATALOG_FILE} must be a JSON array")

    for i, part in enumerate(catalog):
        missing = [f for f in CATALOG_FIELDS if f not in part]
        if missing:
            raise InputError(f"catalog entry {i} is missing {missing}")

    cleaned, seen = [], set()
    for i, r in enumerate(reports):
        for f in ("report_id", "part_id", "operator_note"):
            if not isinstance(r.get(f), str) or not r.get(f):
                raise InputError(f"report {i} has missing or empty {f}")
        rid = r["report_id"]
        if rid in seen:
            raise InputError(f"duplicate report_id {rid}")
        seen.add(rid)
        for f in REPORT_FIELDS:
            if f not in r:
                warnings.append(f"{rid}: field {f} missing, default applied")
        measurements = dict(r.get("measurements") or {})
        for m in MEASUREMENTS:
            if not isinstance(measurements.get(m), (int, float)):
                if m in measurements:
                    warnings.append(f"{rid}: measurement {m} is not numeric, treated as 0")
                measurements[m] = 0.0
        confidence = r.get("confidence", 0)
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise InputError(f"{rid}: confidence must be a number between 0 and 1")
        cleaned.append({
            **r,
            "station": r.get("station") or "UNKNOWN",
            "timestamp": r.get("timestamp") or "",
            "model_finding": r.get("model_finding") or "",
            "confidence": confidence,
            "measurements": measurements,
            "image_count": int(r.get("image_count") or 0),
        })
    known_parts = {p["part_id"] for p in catalog}
    for r in cleaned:
        if r["part_id"] not in known_parts:
            warnings.append(f"{r['report_id']}: part_id {r['part_id']} not in catalog")
    return cleaned, catalog, warnings


def run(ctx):
    reports, catalog, warnings = load_inputs(ctx.input_dir)
    for w in warnings:
        ctx.state.warn(w)
    return StageResult(
        inputs=[ctx.inp(REPORTS_FILE), ctx.inp(CATALOG_FILE)],
        details={"report_count": len(reports), "catalog_parts": len(catalog), "warnings": len(warnings)},
    )
