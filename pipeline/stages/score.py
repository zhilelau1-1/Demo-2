"""PATTERNS_GROUPED -> PRIORITIES_COMPUTED: deterministic scoring in code (no LLM). Runs after pattern
grouping because pattern membership contributes to the score; the hashes of the consumed artifacts
are stored as proof of ordering."""

from .. import scoring
from ..artifacts import display_path, sha256_file, utc_now, write_json
from ..context import INTERPRETED_FILE, NORMALISED_FILE, PATTERNS_FILE, SCORES_FILE, StageResult

FORMULA = {
    "base_priority": scoring.BASE_PRIORITY,
    "confidence_bonus": "confidence * 10",
    "evidence_bonus": "image_count * 2",
    "measurement_bonus": {name: f"+{bonus} if >= {threshold}" for name, threshold, bonus in scoring.MEASUREMENT_BONUSES},
    "pattern_membership_bonus": "12 per pattern containing the report",
    "disposition_bonus": scoring.DISPOSITION_BONUS,
}


def run(ctx):
    inputs = [ctx.out(NORMALISED_FILE), ctx.out(INTERPRETED_FILE), ctx.out(PATTERNS_FILE)]
    scores = scoring.compute_scores(ctx.load(NORMALISED_FILE), ctx.load(INTERPRETED_FILE), ctx.load(PATTERNS_FILE))
    write_json(ctx.out(SCORES_FILE), {
        "formula_version": scoring.FORMULA_VERSION,
        "formula": FORMULA,
        "tie_break": scoring.TIE_BREAK,
        "computed_at": utc_now(),
        "computed_from": {display_path(p): sha256_file(p) for p in inputs},
        "scores": scores,
    })
    return StageResult(
        inputs=inputs,
        outputs=[ctx.out(SCORES_FILE)],
        details={"max_score": scores[0]["priority_score"], "min_score": scores[-1]["priority_score"]},
    )
