"""Independent SYNTHETIC tests only. No model/tool execution or real job mutation."""

import json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skill import native_host as n
from PIL import Image

SPEC = Path(__file__).parent / "fixtures/native-host.figure-spec.json"


class HostFixture:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="synthetic-independent-review-")
        self.root = Path(self.tmp.name)
        self.job = self.root / "job"
        n.prepare(SPEC, self.job)

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, stage, data):
        p = self.root / (stage + ".json")
        p.write_text(json.dumps(data))
        return n.record(self.job, stage, p, "SYNTHETIC_TEST_NOT_LIVE")

    def first(self):
        self.record(
            "Retriever",
            {
                "status": "succeeded",
                "observations": "synthetic",
                "source_paths": ["synthetic"],
            },
        )

    def full(self):
        self.first()
        self.record(
            "Planner", {"status": "succeeded", "design_description": "synthetic"}
        )
        self.record(
            "Stylist",
            {
                "status": "succeeded",
                "design_description": "synthetic",
                "prompt": "synthetic only",
            },
        )
        p = self.root / "fixture.png"
        Image.new("RGB", (2, 2)).save(p)
        n.record(
            self.job,
            "Visualizer",
            p,
            "SYNTHETIC_TEST_NOT_LIVE",
            n.next_task(self.job)["prompt_sha256"],
        )
        self.record(
            "Critic",
            {
                "status": "succeeded",
                "image_sha256": n.sha(p.read_bytes()),
                "verdict": "revise",
                "observed_scientific_fidelity": "synthetic",
                "aesthetics": "synthetic",
                "issues": ["synthetic"],
            },
        )

    def mutate(self, name, key, value):
        p = self.job / name
        d = json.loads(p.read_text())
        d[key] = value
        p.write_text(json.dumps(d))


class Guards(HostFixture, unittest.TestCase):
    def test_failed_stage_rejected(self):
        with self.assertRaises(AssertionError):
            self.record("Retriever", {"status": "failed", "error": "synthetic failure"})
        self.assertEqual(n.next_task(self.job)["stage"], "Retriever")

    def test_verdict_receipt_mismatch_rejected(self):
        self.full()
        self.mutate("Critic.json", "verdict", "accept")
        with self.assertRaises(AssertionError):
            n.next_task(self.job)

    def test_empty_artifact_set_rejected(self):
        self.first()
        self.mutate("Retriever.json", "artifacts", {})
        with self.assertRaises(AssertionError):
            n.next_task(self.job)

    def test_missing_host_task_rejected(self):
        self.first()
        self.mutate("Retriever.json", "host_task_id", "")
        with self.assertRaises(AssertionError):
            n.next_task(self.job)

    def test_broken_chain_rejected(self):
        self.full()
        self.mutate("Planner.json", "previous_receipt_sha256", "bad")
        with self.assertRaises(AssertionError):
            n.next_task(self.job)

    def test_prompt_receipt_mismatch_rejected(self):
        self.full()
        self.mutate("Visualizer.json", "prompt_sha256", "bad")
        with self.assertRaises(AssertionError):
            n.next_task(self.job)

    def test_incomplete_success_payload_rejected(self):
        with self.assertRaises(AssertionError):
            self.record("Retriever", {"status": "succeeded"})

    def test_no_overwrite_or_implicit_revision(self):
        self.full()
        self.assertEqual(n.next_task(self.job)["status"], "REVISION_REQUIRED")
        with self.assertRaises(AssertionError):
            self.record("Critic", {"status": "succeeded"})


if __name__ == "__main__":
    unittest.main()
