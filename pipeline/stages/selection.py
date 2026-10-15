"""PRIORITIES_COMPUTED -> CASES_SELECTED: flag the top-N reports from the computed scores."""

from .. import scoring
from ..artifacts import display_path, sha256_file, write_json
from ..context import FLAGGED_FILE, SCORES_FILE, StageResult


def run(ctx):
    scores = ctx.load(SCORES_FILE)["scores"]
    top_n = min(ctx.top_n, len(scores))
    flagged = scoring.select_top(scores, top_n)
    by_id = {s["report_id"]: s for s in scores}
    write_json(ctx.out(FLAGGED_FILE), {
        "top_n": top_n,
        "selected_from": {display_path(ctx.out(SCORES_FILE)): sha256_file(ctx.out(SCORES_FILE))},
        "tie_break": scoring.TIE_BREAK,
        "flagged_report_ids": flagged,
        "cases": [{"rank": by_id[r]["rank"], "report_id": r, "priority_score": by_id[r]["priority_score"]} for r in flagged],
    })
    return StageResult(inputs=[ctx.out(SCORES_FILE)], outputs=[ctx.out(FLAGGED_FILE)], details={"flagged_report_ids": flagged})
