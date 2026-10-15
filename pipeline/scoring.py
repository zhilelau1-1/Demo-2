"""Deterministic priority scoring. No LLM involvement: the same interpreted reports and patterns
always yield the same scores, ranks and top-N selection.

Decimal arithmetic avoids float artefacts such as 0.82 * 10 = 8.200000000000001.
"""

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

FORMULA_VERSION = "v1"

BASE_PRIORITY = {"critical": 40, "high": 25, "medium": 10, "low": 3}
DISPOSITION_BONUS = {"hold": 12, "reinspect": 6, "monitor": 2, "pass": 0}
CONFIDENCE_MULTIPLIER = 10
EVIDENCE_PER_IMAGE = 2
PATTERN_MEMBERSHIP_BONUS = 12
# (measurement, threshold, bonus) - each applies independently, regardless of defect category.
MEASUREMENT_BONUSES = (
    ("void_ratio", "0.20", 15),
    ("misalignment_mm", "0.20", 10),
    ("scratch_length_mm", "6.0", 8),
)
TIE_BREAK = "priority_score descending, then timestamp ascending, then report_id ascending"

_CENT = Decimal("0.01")


def _dec(value):
    return Decimal(str(value))


def _money(value):
    return float(_dec(value).quantize(_CENT, rounding=ROUND_HALF_UP))


def score_report(severity, disposition, confidence, image_count, measurements, pattern_ids):
    """Return the full score breakdown for one report."""
    base = BASE_PRIORITY[severity]
    confidence_bonus = _dec(confidence) * CONFIDENCE_MULTIPLIER
    evidence_bonus = int(image_count) * EVIDENCE_PER_IMAGE
    hits = []
    measurement_bonus = 0
    for name, threshold, bonus in MEASUREMENT_BONUSES:
        value = measurements.get(name, 0) or 0
        if _dec(value) >= Decimal(threshold):
            measurement_bonus += bonus
            hits.append({"measurement": name, "value": value, "threshold": float(threshold), "bonus": bonus})
    unique_patterns = sorted(set(pattern_ids))
    pattern_bonus = PATTERN_MEMBERSHIP_BONUS * len(unique_patterns)
    disposition_bonus = DISPOSITION_BONUS[disposition]
    total = (
        Decimal(base) + confidence_bonus + Decimal(evidence_bonus) + Decimal(measurement_bonus)
        + Decimal(pattern_bonus) + Decimal(disposition_bonus)
    )
    return {
        "components": {
            "base_priority": base,
            "confidence_bonus": _money(confidence_bonus),
            "evidence_bonus": evidence_bonus,
            "measurement_bonus": measurement_bonus,
            "pattern_membership_bonus": pattern_bonus,
            "disposition_bonus": disposition_bonus,
        },
        "measurement_hits": hits,
        "pattern_ids": unique_patterns,
        "priority_score": _money(total),
    }


def timestamp_key(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("inf")


def compute_scores(normalised, interpreted, patterns):
    """Score every report and return records sorted by rank (1 = highest priority)."""
    interp_by_id = {r["report_id"]: r for r in interpreted}
    membership = {}
    for p in patterns:
        for rid in p["supporting_report_ids"]:
            membership.setdefault(rid, []).append(p["pattern_id"])
    scored = []
    for rec in normalised:
        rid = rec["report_id"]
        interp = interp_by_id[rid]
        breakdown = score_report(
            interp["severity"], interp["disposition"], rec["confidence"], rec["image_count"],
            rec["measurements"], membership.get(rid, []),
        )
        scored.append({
            "report_id": rid,
            "part_id": rec["part_id"],
            "part_name": rec["part_name"],
            "station": rec["station"],
            "timestamp": rec["timestamp"],
            "defect_category": interp["defect_category"],
            "severity": interp["severity"],
            "disposition": interp["disposition"],
            "confidence": rec["confidence"],
            "image_count": rec["image_count"],
            "measurements": rec["measurements"],
            **breakdown,
        })
    scored.sort(key=lambda s: (-s["priority_score"], timestamp_key(s["timestamp"]), s["report_id"]))
    for rank, s in enumerate(scored, start=1):
        s["rank"] = rank
    return scored


def select_top(scored, top_n):
    """Top-N report IDs from an already ranked list."""
    return [s["report_id"] for s in sorted(scored, key=lambda s: s["rank"])[:top_n]]
