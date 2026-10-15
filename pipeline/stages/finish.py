"""EXPLANATIONS_GENERATED -> VALIDATION_COMPLETE -> RESULTS_FINALISED."""

from ..artifacts import utc_now, write_json
from ..checks import run_checks
from ..context import VALIDATION_FILE, StageResult


class ValidationFailed(RuntimeError):
    pass


def validate(ctx):
    results = run_checks(ctx.input_dir, ctx.out_dir, require_finalised=False)
    report = {
        "checked_at": utc_now(),
        "passed": all(r.passed for r in results),
        "checks": [{"name": r.name, "passed": r.passed, "problems": r.problems} for r in results],
    }
    write_json(ctx.out(VALIDATION_FILE), report)
    if not report["passed"]:
        failed = [f"{r.name}: {r.problems[:3]}" for r in results if not r.passed]
        raise ValidationFailed("validation failed:\n  " + "\n  ".join(failed))
    return StageResult(outputs=[ctx.out(VALIDATION_FILE)], details={"checks_passed": len(results)})


def finalise(ctx):
    return StageResult(details={"finalised_at": utc_now()})
