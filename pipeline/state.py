"""Pipeline state machine. Stages may only advance one step at a time, in the fixed order below.

Every transition is recorded in run_manifest.json together with the sha256 of the artifacts the
stage read and wrote, which is what makes a run auditable and resumable.
"""

from enum import Enum
from pathlib import Path

from . import PIPELINE_VERSION
from .artifacts import display_path, read_json, sha256_file, utc_now, write_json


class Stage(str, Enum):
    INIT = "INIT"
    INPUTS_LOADED = "INPUTS_LOADED"
    REPORTS_NORMALISED = "REPORTS_NORMALISED"
    REPORTS_INTERPRETED = "REPORTS_INTERPRETED"
    PATTERNS_GROUPED = "PATTERNS_GROUPED"
    PRIORITIES_COMPUTED = "PRIORITIES_COMPUTED"
    CASES_SELECTED = "CASES_SELECTED"
    ROUTING_COMPLETE = "ROUTING_COMPLETE"
    EXPLANATIONS_GENERATED = "EXPLANATIONS_GENERATED"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    RESULTS_FINALISED = "RESULTS_FINALISED"


STAGE_ORDER = list(Stage)


def stage_index(stage):
    return STAGE_ORDER.index(Stage(stage))


class PipelineStateError(RuntimeError):
    pass


def _artifact_hashes(paths):
    return {display_path(p): sha256_file(p) for p in paths}


class PipelineState:
    def __init__(self, manifest_path, run_id, llm_info, input_paths):
        self.manifest_path = Path(manifest_path)
        self.manifest = {
            "run_id": run_id,
            "pipeline_version": PIPELINE_VERSION,
            "started_at": utc_now(),
            "status": "running",
            "llm": llm_info,
            "inputs": _artifact_hashes(input_paths),
            "stage_order": [s.value for s in STAGE_ORDER],
            "stages": [{"stage": Stage.INIT.value, "completed_at": utc_now(), "inputs": {}, "outputs": {}}],
            "resumes": [],
            "warnings": [],
        }
        self.save()

    @property
    def run_id(self):
        return self.manifest["run_id"]

    @property
    def current(self):
        return Stage(self.manifest["stages"][-1]["stage"])

    def require(self, stage):
        if self.current != Stage(stage):
            raise PipelineStateError(f"expected pipeline state {Stage(stage).value}, current state is {self.current.value}")

    def require_next(self, target):
        """Refuse to run a stage unless it is exactly the next one in the fixed order."""
        target = Stage(target)
        idx = stage_index(self.current)
        if idx + 1 >= len(STAGE_ORDER) or STAGE_ORDER[idx + 1] != target:
            raise PipelineStateError(f"cannot move from {self.current.value} to {target.value}: stages must run in order")

    def advance(self, target, inputs=(), outputs=(), details=None):
        self.require_next(target)
        entry = {
            "stage": Stage(target).value,
            "completed_at": utc_now(),
            "inputs": _artifact_hashes(inputs),
            "outputs": _artifact_hashes(outputs),
        }
        if details:
            entry["details"] = details
        self.manifest["stages"].append(entry)
        self.save()

    def warn(self, message):
        self.manifest["warnings"].append(message)
        self.save()

    def finish(self, status, error=None):
        self.manifest["status"] = status
        self.manifest["finished_at"] = utc_now()
        if error:
            self.manifest["error"] = error
        self.save()

    def save(self):
        write_json(self.manifest_path, self.manifest)

    @classmethod
    def resume(cls, manifest_path, from_stage, input_paths):
        """Reload a previous run and rewind it so that `from_stage` is the next stage to run.

        Every artifact recorded by the kept stages must still match its recorded hash; otherwise the
        replay would mix outputs from different runs.
        """
        from_stage = Stage(from_stage)
        if from_stage == Stage.INIT:
            raise PipelineStateError("cannot resume from INIT; run the full pipeline instead")
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise PipelineStateError(f"no manifest at {manifest_path} to resume from")
        manifest = read_json(manifest_path)
        keep = [s for s in manifest["stages"] if stage_index(s["stage"]) < stage_index(from_stage)]
        if not keep or Stage(keep[-1]["stage"]) != STAGE_ORDER[stage_index(from_stage) - 1]:
            raise PipelineStateError(f"previous run never reached {STAGE_ORDER[stage_index(from_stage) - 1].value}")
        # Artifacts are matched by file name inside the current input/output directories, so a run
        # directory can be moved or copied and still be resumed.
        by_name = {Path(p).name: Path(p) for p in input_paths}
        current_inputs = {name: sha256_file(p) for name, p in by_name.items()}
        if current_inputs != {Path(p).name: h for p, h in manifest["inputs"].items()}:
            raise PipelineStateError("input fixtures changed since the recorded run; run the full pipeline")
        for entry in keep:
            for recorded, digest in {**entry["inputs"], **entry["outputs"]}.items():
                name = Path(recorded).name
                path = by_name.get(name, manifest_path.parent / name)
                if not path.exists() or sha256_file(path) != digest:
                    raise PipelineStateError(f"artifact {name} changed since stage {entry['stage']}; cannot resume")
        self = cls.__new__(cls)
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.manifest["stages"] = keep
        self.manifest["status"] = "running"
        self.manifest.pop("finished_at", None)
        self.manifest.pop("error", None)
        self.manifest["resumes"].append({"at": utc_now(), "from_stage": from_stage.value})
        self.save()
        return self
