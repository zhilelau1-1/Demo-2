"""Artifact validation. Used by the pipeline's VALIDATION_COMPLETE stage and by validate.py.

Every check re-derives its expectation from the artifacts on disk (for example, scores are recomputed
from interpreted_reports.json + patterns.json) rather than trusting what the pipeline claims.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import scoring, vocab
from .artifacts import read_json, sha256_file
from .context import (
    ALERT_FILE, CATALOG_FILE, DISAGREEMENTS_FILE, FLAGGED_FILE, INTERPRETED_FILE, LLM_LOG_FILE, MANIFEST_FILE,
    NORMALISED_FILE, PATTERNS_FILE, REINSPECTION_FILE, REPORTS_FILE, ROUTING_FILE, SCORES_FILE, SUMMARIES_FILE,
    TRENDS_FILE,
)
from .state import STAGE_ORDER, Stage

REQUIRED_LLM_STAGES = ("normalisation_translation", "report_interpretation", "pattern_grouping", "case_routing")
LLM_RECORD_FIELDS = ("stage", "timestamp", "provider", "model", "prompt_hash", "input_artifacts", "output_artifact")
OUTPUT_JSON = (NORMALISED_FILE, INTERPRETED_FILE, PATTERNS_FILE, SCORES_FILE, FLAGGED_FILE, ROUTING_FILE,
               TRENDS_FILE, DISAGREEMENTS_FILE, ALERT_FILE, MANIFEST_FILE)
OUTPUT_TEXT = (SUMMARIES_FILE, REINSPECTION_FILE, LLM_LOG_FILE)


@dataclass
class CheckResult:
    name: str
    passed: bool = True
    problems: list = field(default_factory=list)

    def fail(self, message):
        self.passed = False
        self.problems.append(message)


class Unavailable(Exception):
    pass


class Artifacts:
    """Lazy, cached access to artifacts; a missing or invalid file raises Unavailable."""

    def __init__(self, input_dir, out_dir):
        self.input_dir, self.out_dir = Path(input_dir), Path(out_dir)
        self._cache = {}

    def path(self, name):
        return (self.input_dir if name in (REPORTS_FILE, CATALOG_FILE) else self.out_dir) / name

    def json(self, name):
        if name not in self._cache:
            try:
                self._cache[name] = read_json(self.path(name))
            except (OSError, json.JSONDecodeError) as e:
                raise Unavailable(f"{name} unavailable: {e}")
        return self._cache[name]

    def text(self, name):
        try:
            return self.path(name).read_text(encoding="utf-8")
        except OSError as e:
            raise Unavailable(f"{name} unavailable: {e}")

    def llm_records(self):
        if "__llm" not in self._cache:
            records = []
            for n, line in enumerate(self.text(LLM_LOG_FILE).splitlines(), start=1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise Unavailable(f"{LLM_LOG_FILE} line {n} is not valid JSON: {e}")
            self._cache["__llm"] = records
        return self._cache["__llm"]

    def sha(self, name):
        return sha256_file(self.path(name))


def _manifest_stage(manifest, stage):
    return next((s for s in manifest["stages"] if s["stage"] == stage), None)


def _hash_for(mapping, filename):
    return next((h for p, h in mapping.items() if Path(p).name == filename), None)


def check_artifacts_exist(a, c):
    for name in (REPORTS_FILE, CATALOG_FILE, *OUTPUT_JSON, *OUTPUT_TEXT):
        if not a.path(name).exists():
            c.fail(f"missing {name}")


def check_json_valid(a, c):
    for name in (REPORTS_FILE, CATALOG_FILE, *OUTPUT_JSON):
        try:
            a.json(name)
        except Unavailable as e:
            c.fail(str(e))
    a.llm_records()


def check_all_reports_processed(a, c):
    expected = sorted(r["report_id"] for r in a.json(REPORTS_FILE))
    for name, ids in (
        (NORMALISED_FILE, [r["report_id"] for r in a.json(NORMALISED_FILE)]),
        (INTERPRETED_FILE, [r["report_id"] for r in a.json(INTERPRETED_FILE)]),
        (SCORES_FILE, [r["report_id"] for r in a.json(SCORES_FILE)["scores"]]),
    ):
        if sorted(ids) != expected:
            c.fail(f"{name} covers {sorted(ids)}, expected {expected}")


def check_catalog_join(a, c):
    parts = {p["part_id"]: p for p in a.json(CATALOG_FILE)}
    for r in a.json(NORMALISED_FILE):
        part = parts.get(r["part_id"])
        expected = (part["part_name"], part["product_line"]) if part else ("UNKNOWN", "UNKNOWN")
        if (r["part_name"], r["product_line"]) != expected:
            c.fail(f"{r['report_id']}: part metadata {r['part_name']}/{r['product_line']} does not match catalog {expected}")


def check_translation(a, c):
    raw = {r["report_id"]: r["operator_note"] for r in a.json(REPORTS_FILE)}
    interp = {r["report_id"]: r for r in a.json(INTERPRETED_FILE)}
    for r in a.json(NORMALISED_FILE):
        rid = r["report_id"]
        for f in ("original_operator_note", "note_for_interpretation", "original_language", "translated"):
            if f not in r:
                c.fail(f"{rid}: normalised record missing {f}")
        if r.get("original_operator_note") != raw.get(rid):
            c.fail(f"{rid}: original_operator_note does not preserve the input note verbatim")
        if r.get("original_language") != "en":
            if r.get("translated") is not True:
                c.fail(f"{rid}: non-English note ({r.get('original_language')}) was not translated")
            elif not r.get("note_for_interpretation", "").strip() or r["note_for_interpretation"] == r["original_operator_note"]:
                c.fail(f"{rid}: translated note_for_interpretation is empty or identical to the original")
        elif r.get("translated") is not False:
            c.fail(f"{rid}: English note marked translated")
        i = interp.get(rid, {})
        if (i.get("translated"), i.get("original_language")) != (r.get("translated"), r.get("original_language")):
            c.fail(f"{rid}: interpreted translated/original_language differ from normalisation")


def check_vocabularies(a, c):
    for r in a.json(INTERPRETED_FILE):
        for f in ("defect_category", "severity", "disposition"):
            err = vocab.vocab_error(f, r.get(f))
            if err:
                c.fail(f"interpreted {r['report_id']}: {err}")
    for p in a.json(PATTERNS_FILE):
        err = vocab.vocab_error("pattern_strength", p.get("pattern_strength"))
        if err:
            c.fail(f"pattern {p.get('pattern_id')}: {err}")
    for r in a.json(ROUTING_FILE):
        for t in r.get("teams", []):
            err = vocab.vocab_error("team", t)
            if err:
                c.fail(f"routing {r['report_id']}: {err}")
    err = vocab.vocab_error("alert_rating", a.json(ALERT_FILE).get("rating"))
    if err:
        c.fail(f"alert: {err}")


def check_patterns_use_interpreted(a, c):
    interp = {r["report_id"]: r for r in a.json(INTERPRETED_FILE)}
    ids = set(interp)
    for p in a.json(PATTERNS_FILE):
        members = p.get("supporting_report_ids", [])
        if len(set(members)) < 2 or not set(members) <= ids:
            c.fail(f"pattern {p.get('pattern_id')}: supporting_report_ids {members} invalid")
    success = [r for r in a.llm_records() if r.get("stage") == "pattern_grouping" and r.get("status") == "success"]
    if not success:
        c.fail("no successful pattern_grouping LLM call logged")
        return
    if not any(Path(p).name == INTERPRETED_FILE for p in success[-1]["input_artifacts"]):
        c.fail("pattern_grouping call did not list interpreted_reports.json as an input artifact")
    payload_path = a.out_dir / "llm_io" / "pattern_grouping.input.json"
    if not payload_path.exists():
        c.fail("pattern_grouping input payload was not preserved")
        return
    payload = read_json(payload_path)["reports"]
    if sorted(r["report_id"] for r in payload) != sorted(ids):
        c.fail("pattern_grouping payload does not cover every interpreted report")
    for r in payload:
        i = interp.get(r["report_id"], {})
        for f in ("defect_category", "severity", "disposition"):
            if r.get(f) != i.get(f):
                c.fail(f"pattern_grouping payload {r['report_id']}.{f}={r.get(f)!r} differs from interpreted {i.get(f)!r}")
        for f in ("part_name", "station", "timestamp", "confidence", "measurements", "note_for_interpretation"):
            if f not in r:
                c.fail(f"pattern_grouping payload {r['report_id']} missing {f}")


def check_scores(a, c):
    stored = a.json(SCORES_FILE)
    recomputed = scoring.compute_scores(a.json(NORMALISED_FILE), a.json(INTERPRETED_FILE), a.json(PATTERNS_FILE))
    fields = ("report_id", "rank", "priority_score", "components", "pattern_ids")
    if [{f: s[f] for f in fields} for s in stored["scores"]] != [{f: s[f] for f in fields} for s in recomputed]:
        c.fail("stored priority scores differ from a deterministic recomputation")
    if _hash_for(stored.get("computed_from", {}), PATTERNS_FILE) != a.sha(PATTERNS_FILE):
        c.fail("priority_scores.json was not computed from the current patterns.json")
    if _hash_for(stored.get("computed_from", {}), INTERPRETED_FILE) != a.sha(INTERPRETED_FILE):
        c.fail("priority_scores.json was not computed from the current interpreted_reports.json")
    manifest = a.json(MANIFEST_FILE)
    grouped = _manifest_stage(manifest, Stage.PATTERNS_GROUPED.value)
    computed = _manifest_stage(manifest, Stage.PRIORITIES_COMPUTED.value)
    if not grouped or not computed:
        c.fail("manifest does not record both PATTERNS_GROUPED and PRIORITIES_COMPUTED")
    elif computed["completed_at"] < grouped["completed_at"]:
        c.fail("PRIORITIES_COMPUTED completed before PATTERNS_GROUPED")
    elif _hash_for(grouped["outputs"], PATTERNS_FILE) != _hash_for(computed["inputs"], PATTERNS_FILE):
        c.fail("scoring consumed a different patterns.json than pattern grouping produced")


def check_selection(a, c):
    scores = a.json(SCORES_FILE)["scores"]
    flagged = a.json(FLAGGED_FILE)
    expected = scoring.select_top(scoring.compute_scores(a.json(NORMALISED_FILE), a.json(INTERPRETED_FILE), a.json(PATTERNS_FILE)), min(5, len(scores)))
    if flagged.get("flagged_report_ids") != expected:
        c.fail(f"flagged cases {flagged.get('flagged_report_ids')} != top {len(expected)} by computed score {expected}")
    if _hash_for(flagged.get("selected_from", {}), SCORES_FILE) != a.sha(SCORES_FILE):
        c.fail("flagged_cases.json was not selected from the current priority_scores.json")


def check_routing(a, c):
    flagged = a.json(FLAGGED_FILE)["flagged_report_ids"]
    routing = a.json(ROUTING_FILE)
    ids = [r["report_id"] for r in routing]
    if sorted(ids) != sorted(flagged):
        c.fail(f"routed cases {ids} != flagged cases {flagged}")
    for r in routing:
        if not r.get("teams"):
            c.fail(f"{r['report_id']}: no teams")
        if len(set(r.get("teams", []))) != len(r.get("teams", [])):
            c.fail(f"{r['report_id']}: duplicate teams")
        if not (r.get("briefing_note") or "").strip():
            c.fail(f"{r['report_id']}: empty briefing note")


def _sections(markdown):
    parts = re.split(r"^## ", markdown, flags=re.MULTILINE)
    return parts[1:]


def check_summaries(a, c):
    md = a.text(SUMMARIES_FILE)
    sections = _sections(md)
    scores = {s["report_id"]: s for s in a.json(SCORES_FILE)["scores"]}
    for rid in a.json(FLAGGED_FILE)["flagged_report_ids"]:
        section = next((s for s in sections if re.search(rf"\b{re.escape(rid)}\b", s.splitlines()[0])), None)
        if section is None:
            c.fail(f"{rid}: no section in review_summaries.md")
            continue
        s = scores[rid]
        for label, value in (("part name", s["part_name"]), ("category", s["defect_category"]), ("severity", s["severity"]),
                             ("disposition", s["disposition"]), ("priority score", str(s["priority_score"])),
                             ("pattern membership", "Recurring pattern"), ("explanation", "Why it was prioritised")):
            if value not in section:
                c.fail(f"{rid}: summary missing {label} ({value})")


def check_gate_notes(a, c):
    sections = _sections(a.text(SUMMARIES_FILE))
    for rid in a.json(FLAGGED_FILE)["flagged_report_ids"]:
        section = next((s for s in sections if re.search(rf"\b{re.escape(rid)}\b", s.splitlines()[0])), "")
        m = re.search(r"\*\*Human review gate:\*\*\s*(.+)", section)
        if not m or len(m.group(1).strip()) < 40:
            c.fail(f"{rid}: missing or empty human review gate note")
        elif "before any production decision" not in m.group(1).lower():
            c.fail(f"{rid}: gate note does not state what must be verified before a production decision")


def check_llm_log(a, c):
    records = a.llm_records()
    manifest = a.json(MANIFEST_FILE)
    for n, r in enumerate(records, start=1):
        missing = [f for f in LLM_RECORD_FIELDS if f not in r]
        if missing:
            c.fail(f"record {n} missing {missing}")
            continue
        try:
            datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
        except ValueError:
            c.fail(f"record {n} timestamp {r['timestamp']!r} is not ISO-8601")
        if r.get("run_id") != manifest["run_id"]:
            c.fail(f"record {n} belongs to run {r.get('run_id')}, not {manifest['run_id']}")
    explained = _manifest_stage(manifest, Stage.EXPLANATIONS_GENERATED.value) or {}
    expected = [*REQUIRED_LLM_STAGES, *explained.get("details", {}).get("llm_stages", ["explanation_generation"])]
    for stage in expected:
        ok = [r for r in records if r.get("stage") == stage and r.get("status") == "success"]
        if len(ok) != 1:
            c.fail(f"stage {stage}: expected exactly one successful call record, found {len(ok)}")
    first = {s: next((r["timestamp"] for r in records if r.get("stage") == s and r.get("status") == "success"), None) for s in REQUIRED_LLM_STAGES}
    times = [first[s] for s in REQUIRED_LLM_STAGES]
    if None not in times and times != sorted(times):
        c.fail(f"required LLM stages ran out of order: {dict(zip(REQUIRED_LLM_STAGES, times))}")


def check_stage_sequence(a, c, require_finalised):
    manifest = a.json(MANIFEST_FILE)
    states = [s["stage"] for s in manifest["stages"]]
    if states != [s.value for s in STAGE_ORDER[:len(states)]]:
        c.fail(f"recorded stages {states} are not the required order")
    times = [s["completed_at"] for s in manifest["stages"]]
    if times != sorted(times):
        c.fail("stage completion times are not monotonic")
    last = Stage.RESULTS_FINALISED if require_finalised else Stage.EXPLANATIONS_GENERATED
    if Stage.EXPLANATIONS_GENERATED.value not in states:
        c.fail("pipeline did not reach EXPLANATIONS_GENERATED")
    if require_finalised and (states[-1] != last.value or manifest.get("status") != "complete"):
        c.fail(f"run is not finalised (last stage {states[-1]}, status {manifest.get('status')})")


def check_optional_artifacts(a, c):
    trends = a.json(TRENDS_FILE)
    for key in ("hourly_distribution", "category_distribution", "pattern_trends", "most_affected", "narrative"):
        if key not in trends:
            c.fail(f"shift_trends.json missing {key}")
    report_ids = {r["report_id"] for r in a.json(REPORTS_FILE)}
    for item in a.json(DISAGREEMENTS_FILE):
        if item.get("report_id") not in report_ids or not item.get("disagreement_reason") or not item.get("recommended_follow_up"):
            c.fail(f"disagreement item invalid: {item.get('report_id')}")
    alert = a.json(ALERT_FILE)
    for key in ("rating", "justification", "operational_posture", "evidence"):
        if not alert.get(key):
            c.fail(f"alert_rating.json missing {key}")
    plan = a.text(REINSPECTION_FILE)
    for heading in ("Parts to sample first", "Stations to prioritise", "Defect signatures to watch", "Stop / continue criteria", "Owner team"):
        if heading not in plan:
            c.fail(f"reinspection_plan.md missing section '{heading}'")


CHECKS = [
    ("required artifacts exist", check_artifacts_exist),
    ("JSON artifacts are valid", check_json_valid),
    ("all reports processed", check_all_reports_processed),
    ("reports joined with part catalog", check_catalog_join),
    ("non-English notes preserved and translated", check_translation),
    ("controlled vocabularies respected", check_vocabularies),
    ("pattern grouping used interpreted reports", check_patterns_use_interpreted),
    ("priority scores deterministic and computed after grouping", check_scores),
    ("top 5 selected from computed scores", check_selection),
    ("routing covers flagged cases with valid teams", check_routing),
    ("every flagged case summarised", check_summaries),
    ("every flagged case has a human review gate note", check_gate_notes),
    ("LLM call log has separate records per stage", check_llm_log),
    ("pipeline stages ran in the required order", None),
    ("optional analysis artifacts are well-formed", check_optional_artifacts),
]


def run_checks(input_dir, out_dir, require_finalised=True):
    a = Artifacts(input_dir, out_dir)
    results = []
    for name, fn in CHECKS:
        c = CheckResult(name)
        try:
            if fn is None:
                check_stage_sequence(a, c, require_finalised)
            else:
                fn(a, c)
        except Unavailable as e:
            c.fail(str(e))
        except (KeyError, TypeError, IndexError, ValueError) as e:
            c.fail(f"malformed artifact: {type(e).__name__}: {e}")
        results.append(c)
    return results
