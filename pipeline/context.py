"""Shared run context and helpers for assembling per-case views from stage artifacts."""

from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import read_json

REPORTS_FILE = "inspection_reports.json"
CATALOG_FILE = "parts_catalog.json"
NORMALISED_FILE = "normalised_reports.json"
INTERPRETED_FILE = "interpreted_reports.json"
PATTERNS_FILE = "patterns.json"
SCORES_FILE = "priority_scores.json"
FLAGGED_FILE = "flagged_cases.json"
ROUTING_FILE = "case_routing.json"
SUMMARIES_FILE = "review_summaries.md"
TRENDS_FILE = "shift_trends.json"
DISAGREEMENTS_FILE = "disagreement_cases.json"
ALERT_FILE = "alert_rating.json"
REINSPECTION_FILE = "reinspection_plan.md"
MANIFEST_FILE = "run_manifest.json"
VALIDATION_FILE = "validation_report.json"
LLM_LOG_FILE = "llm_calls.jsonl"


@dataclass
class StageResult:
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class Context:
    input_dir: Path
    out_dir: Path
    llm: object
    state: object
    top_n: int = 5

    def inp(self, name):
        return self.input_dir / name

    def out(self, name):
        return self.out_dir / name

    def load(self, name):
        return read_json(self.out(name))


def build_cases(normalised, interpreted, scores):
    """Join normalised, interpreted and scored records into one view per report, keyed by report_id."""
    norm = {r["report_id"]: r for r in normalised}
    interp = {r["report_id"]: r for r in interpreted}
    cases = {}
    for s in scores:
        rid = s["report_id"]
        n, i = norm[rid], interp[rid]
        cases[rid] = {
            "rank": s["rank"],
            "report_id": rid,
            "part_id": n["part_id"],
            "part_name": n["part_name"],
            "product_line": n["product_line"],
            "station": n["station"],
            "timestamp": n["timestamp"],
            "model_finding": n["model_finding"],
            "note_for_interpretation": n["note_for_interpretation"],
            "original_language": n["original_language"],
            "translated": n["translated"],
            "defect_category": i["defect_category"],
            "severity": i["severity"],
            "disposition": i["disposition"],
            "reason_short": i["reason_short"],
            "confidence": n["confidence"],
            "image_count": n["image_count"],
            "measurements": n["measurements"],
            "priority_score": s["priority_score"],
            "score_components": s["components"],
            "measurement_hits": s["measurement_hits"],
            "pattern_ids": s["pattern_ids"],
        }
    return cases


def flagged_cases(ctx):
    """Flagged cases in rank order, fully enriched."""
    scores = ctx.load(SCORES_FILE)["scores"]
    cases = build_cases(ctx.load(NORMALISED_FILE), ctx.load(INTERPRETED_FILE), scores)
    flagged = ctx.load(FLAGGED_FILE)["flagged_report_ids"]
    return [cases[rid] for rid in flagged]
