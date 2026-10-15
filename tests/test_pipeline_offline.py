"""End-to-end runs with the offline provider, on the public fixture and on perturbed fixtures."""

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline.artifacts import read_json, write_json
from pipeline.checks import run_checks
from pipeline.run import main

ROOT = Path(__file__).resolve().parent.parent


def run_pipeline(work, *extra):
    return main(["--input-dir", str(work), "--out-dir", str(work), "--llm-mode", "offline", "--cache-dir", str(work / ".cache"), *extra])


def failed_checks(work):
    return {r.name: r.problems for r in run_checks(work, work) if not r.passed}


class PublicFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp())
        for name in ("inspection_reports.json", "parts_catalog.json"):
            shutil.copy(ROOT / name, cls.work / name)
        cls.exit_code = run_pipeline(cls.work)

    def test_pipeline_and_validation_pass(self):
        self.assertEqual(self.exit_code, 0)
        self.assertEqual(failed_checks(self.work), {})

    def test_r09_translated_from_malay(self):
        r09 = next(r for r in read_json(self.work / "normalised_reports.json") if r["report_id"] == "R09")
        self.assertEqual(r09["original_language"], "ms")
        self.assertTrue(r09["translated"])
        self.assertIn("penjajaran", r09["original_operator_note"])
        self.assertNotIn("penjajaran", r09["note_for_interpretation"])

    def test_at_least_two_patterns(self):
        self.assertGreaterEqual(len(read_json(self.work / "patterns.json")), 2)

    def test_separate_llm_records(self):
        stages = [json.loads(l)["stage"] for l in (self.work / "llm_calls.jsonl").read_text().splitlines()]
        for stage in ("report_interpretation", "pattern_grouping", "case_routing"):
            self.assertEqual(stages.count(stage), 1)

    def test_tampered_patterns_fail_validation(self):
        work = Path(tempfile.mkdtemp())
        shutil.copytree(self.work, work, dirs_exist_ok=True)
        patterns = read_json(work / "patterns.json")
        patterns[0]["supporting_report_ids"] = patterns[0]["supporting_report_ids"][:1] + ["R10"]
        write_json(work / "patterns.json", patterns)
        failed = failed_checks(work)
        self.assertIn("priority scores deterministic and computed after grouping", failed)

    def test_resume_from_scoring_reuses_earlier_stages(self):
        work = Path(tempfile.mkdtemp())
        shutil.copytree(self.work, work, dirs_exist_ok=True)
        before = (work / "interpreted_reports.json").read_text()
        self.assertEqual(run_pipeline(work, "--from-stage", "PRIORITIES_COMPUTED"), 0)
        self.assertEqual((work / "interpreted_reports.json").read_text(), before)
        self.assertEqual(failed_checks(work), {})
        manifest = read_json(work / "run_manifest.json")
        self.assertEqual(manifest["resumes"][-1]["from_stage"], "PRIORITIES_COMPUTED")


class PerturbedFixtureTest(unittest.TestCase):
    def test_shuffled_renamed_multilingual_fixture(self):
        work = Path(tempfile.mkdtemp())
        reports = read_json(ROOT / "inspection_reports.json")
        random.Random(7).shuffle(reports)
        for n, r in enumerate(reports):
            r["report_id"] = f"INS-{900 + n}"
        reports[0]["operator_note"] = "Inspección visual: los pines del lado izquierdo parecen ligeramente desalineados, posible segunda revision."
        reports[1]["part_id"] = "PT-999"  # not in catalog
        write_json(work / "inspection_reports.json", reports)
        shutil.copy(ROOT / "parts_catalog.json", work / "parts_catalog.json")
        self.assertEqual(run_pipeline(work), 0)
        self.assertEqual(failed_checks(work), {})
        normalised = {r["report_id"]: r for r in read_json(work / "normalised_reports.json")}
        self.assertEqual(normalised[reports[0]["report_id"]]["original_language"], "es")
        self.assertEqual(normalised[reports[1]["report_id"]]["part_name"], "UNKNOWN")

    def test_fewer_than_five_reports(self):
        work = Path(tempfile.mkdtemp())
        write_json(work / "inspection_reports.json", read_json(ROOT / "inspection_reports.json")[:4])
        shutil.copy(ROOT / "parts_catalog.json", work / "parts_catalog.json")
        self.assertEqual(run_pipeline(work), 0)
        self.assertEqual(len(read_json(work / "flagged_cases.json")["flagged_report_ids"]), 4)

    def test_untranslatable_language_is_reported_not_hidden(self):
        work = Path(tempfile.mkdtemp())
        reports = read_json(ROOT / "inspection_reports.json")
        reports[0]["operator_note"] = "焊点J4附近有空洞，比昨天的样品更大。"
        write_json(work / "inspection_reports.json", reports)
        shutil.copy(ROOT / "parts_catalog.json", work / "parts_catalog.json")
        self.assertEqual(run_pipeline(work), 1)  # validation stage fails: offline rules cannot translate zh
        report = read_json(work / "validation_report.json")
        failed = [c["name"] for c in report["checks"] if not c["passed"]]
        self.assertEqual(failed, ["non-English notes preserved and translated"])


if __name__ == "__main__":
    unittest.main()
