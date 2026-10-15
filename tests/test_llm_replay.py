import json
import tempfile
import unittest
from pathlib import Path

from pipeline.llm import LLMClient, LLMError

SCHEMA = {"type": "object"}


def _call(client):
    return client.call(
        stage="shift_trend_summary", pipeline_state="EXPLANATIONS_GENERATED", instructions="Summarise.",
        payload={"report_count": 1, "shift_window": {"start": "a", "end": "b"}, "category_distribution": {},
                 "most_affected": {"station": {"name": "S"}, "part": {"name": "P", "product_line": "L"}}, "pattern_trends": []},
        schema=SCHEMA, validate=lambda d: [], input_artifacts=[], output_artifact="out.json",
    )


class ReplayTest(unittest.TestCase):
    def test_recorded_response_is_replayed_and_miss_fails(self):
        work = Path(tempfile.mkdtemp())
        cache = work / "cache"
        recorder = LLMClient("offline", work, "run-1", cache_dir=cache)
        recorder.resolved_mode = "live"  # record like a live call, using the offline provider as the backend
        recorder.model = "test-model"
        recorder.reset_log()
        first = _call(recorder)

        replayer = LLMClient("replay", work / "replay", "run-2", model="test-model", cache_dir=cache)
        replayer.reset_log()
        self.assertEqual(_call(replayer), first)
        record = json.loads((work / "replay" / "llm_calls.jsonl").read_text().splitlines()[0])
        self.assertEqual(record["mode"], "replay")
        self.assertEqual(record["replayed_from"], "offline-rules")

        other_model = LLMClient("replay", work / "miss", "run-3", model="another-model", cache_dir=cache)
        other_model.reset_log()
        with self.assertRaises(LLMError):
            _call(other_model)


if __name__ == "__main__":
    unittest.main()
