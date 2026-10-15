import tempfile
import unittest
from pathlib import Path

from pipeline import vocab
from pipeline.state import PipelineState, PipelineStateError, Stage
from pipeline.stages.interpret import validate_interpretations
from pipeline.stages.patterns import validate_patterns
from pipeline.stages.route import validate_routes


class StateMachineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.state = PipelineState(self.tmp / "run_manifest.json", "test-run", {}, [])

    def test_advances_in_order(self):
        self.state.advance(Stage.INPUTS_LOADED)
        self.state.advance(Stage.REPORTS_NORMALISED)
        self.assertEqual(self.state.current, Stage.REPORTS_NORMALISED)

    def test_refuses_skipping_a_stage(self):
        with self.assertRaises(PipelineStateError):
            self.state.advance(Stage.PRIORITIES_COMPUTED)

    def test_refuses_repeating_a_stage(self):
        self.state.advance(Stage.INPUTS_LOADED)
        with self.assertRaises(PipelineStateError):
            self.state.advance(Stage.INPUTS_LOADED)

    def test_stage_order_matches_spec(self):
        self.assertEqual([s.value for s in Stage], [
            "INIT", "INPUTS_LOADED", "REPORTS_NORMALISED", "REPORTS_INTERPRETED", "PATTERNS_GROUPED",
            "PRIORITIES_COMPUTED", "CASES_SELECTED", "ROUTING_COMPLETE", "EXPLANATIONS_GENERATED",
            "VALIDATION_COMPLETE", "RESULTS_FINALISED",
        ])


class VocabularyValidationTest(unittest.TestCase):
    def test_rejects_invented_category(self):
        data = {"interpretations": [{"report_id": "R1", "defect_category": "cold_joint", "severity": "high", "disposition": "hold", "reason_short": "x"}]}
        self.assertTrue(any("cold_joint" in e for e in validate_interpretations(data, ["R1"])))

    def test_rejects_missing_report(self):
        data = {"interpretations": [{"report_id": "R1", "defect_category": "uncertain", "severity": "low", "disposition": "pass", "reason_short": "x"}]}
        self.assertTrue(validate_interpretations(data, ["R1", "R2"]))

    def test_accepts_valid(self):
        data = {"interpretations": [{"report_id": "R1", "defect_category": "uncertain", "severity": "low", "disposition": "pass", "reason_short": "x"}]}
        self.assertEqual(validate_interpretations(data, ["R1"]), [])

    def test_rejects_single_report_pattern_and_bad_strength(self):
        data = {"patterns": [{"title": "t", "supporting_report_ids": ["R1"], "pattern_strength": "very strong", "likely_driver": "d", "recommended_action": "a"}]}
        errors = validate_patterns(data, {"R1", "R2"})
        self.assertEqual(len(errors), 2)

    def test_rejects_unknown_team(self):
        data = {"routes": [{"report_id": "R1", "teams": ["Quality", "Logistics"], "briefing_note": "b"}]}
        self.assertTrue(any("Logistics" in e for e in validate_routes(data, ["R1"])))

    def test_vocabularies_match_spec(self):
        self.assertEqual(vocab.OWNER_TEAMS, ("Quality", "Process Engineering", "Manufacturing", "Supplier Quality"))
        self.assertEqual(vocab.DISPOSITIONS, ("hold", "reinspect", "monitor", "pass"))


if __name__ == "__main__":
    unittest.main()
