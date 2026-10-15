import unittest

from pipeline import scoring


class ScoreReportTest(unittest.TestCase):
    def test_hand_computed_example(self):
        # critical 40 + 0.9*10 + 4*2 + void>=0.20 15 + one pattern 12 + hold 12 = 96.0
        b = scoring.score_report("critical", "hold", 0.9, 4, {"void_ratio": 0.24, "misalignment_mm": 0, "scratch_length_mm": 0}, ["P04"])
        self.assertEqual(b["priority_score"], 96.0)
        self.assertEqual(b["components"]["measurement_bonus"], 15)

    def test_confidence_bonus_has_no_float_artefacts(self):
        b = scoring.score_report("medium", "reinspect", 0.82, 3, {}, [])
        self.assertEqual(b["components"]["confidence_bonus"], 8.2)
        self.assertEqual(b["priority_score"], 10 + 8.2 + 6 + 6)

    def test_thresholds_are_inclusive_and_independent(self):
        m = {"void_ratio": 0.20, "misalignment_mm": 0.20, "scratch_length_mm": 6.0}
        b = scoring.score_report("low", "pass", 0, 0, m, [])
        self.assertEqual(b["components"]["measurement_bonus"], 15 + 10 + 8)
        below = scoring.score_report("low", "pass", 0, 0, {"void_ratio": 0.19, "misalignment_mm": 0.199, "scratch_length_mm": 5.99}, [])
        self.assertEqual(below["components"]["measurement_bonus"], 0)

    def test_pattern_bonus_per_distinct_pattern(self):
        b = scoring.score_report("low", "pass", 0, 0, {}, ["P01", "P02", "P01"])
        self.assertEqual(b["components"]["pattern_membership_bonus"], 24)
        self.assertEqual(b["pattern_ids"], ["P01", "P02"])


class RankingTest(unittest.TestCase):
    def _rec(self, rid, ts):
        return {"report_id": rid, "part_id": "X", "part_name": "X", "station": "S", "timestamp": ts,
                "confidence": 0.5, "image_count": 1, "measurements": {}}

    def test_ties_break_on_timestamp_then_id(self):
        normalised = [self._rec("B", "2026-01-01T10:00:00Z"), self._rec("A", "2026-01-01T10:00:00Z"), self._rec("C", "2026-01-01T09:00:00Z")]
        interpreted = [{"report_id": r["report_id"], "severity": "low", "disposition": "monitor", "defect_category": "uncertain"} for r in normalised]
        scored = scoring.compute_scores(normalised, interpreted, [])
        self.assertEqual([s["report_id"] for s in scored], ["C", "A", "B"])
        self.assertEqual(scoring.select_top(scored, 2), ["C", "A"])

    def test_pattern_membership_changes_rank(self):
        normalised = [self._rec("A", "2026-01-01T09:00:00Z"), self._rec("B", "2026-01-01T10:00:00Z")]
        interpreted = [{"report_id": r["report_id"], "severity": "low", "disposition": "monitor", "defect_category": "uncertain"} for r in normalised]
        patterns = [{"pattern_id": "P01", "supporting_report_ids": ["B"]}]
        self.assertEqual(scoring.compute_scores(normalised, interpreted, patterns)[0]["report_id"], "B")


if __name__ == "__main__":
    unittest.main()
