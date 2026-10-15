import json, tempfile, unittest
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skill import native_host as n

SPEC = Path(__file__).parent / "fixtures/native-host.figure-spec.json"


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.job = self.root / "job"
        n.prepare(SPEC, self.job)

    def tearDown(self):
        self.tmp.cleanup()

    def payload(self, stage, data):
        p = self.root / (stage + ".json")
        p.write_text(json.dumps(data))
        return p

    def stage(self, role, data):
        return n.record(
            self.job,
            role,
            self.payload(role, {"status": "succeeded", **data}),
            "SYNTHETIC_TEST_NOT_LIVE",
        )

    def preimage(self):
        self.stage(
            "Retriever",
            {"observations": ["Synthetic fixture"], "source_paths": ["fixture"]},
        )
        self.stage("Planner", {"design_description": "Synthetic fixture"})
        self.stage(
            "Stylist",
            {
                "prompt": "synthetic fixture only",
                "design_description": "Synthetic test, not actual generation",
            },
        )

    def test_initial(self):
        self.assertEqual(n.next_task(self.job)["stage"], "Retriever")

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):
            n.prepare(SPEC, self.job)

    def test_wrong_stage(self):
        with self.assertRaises(AssertionError):
            self.stage("Planner", {"x": 1})

    def test_missing_prompt(self):
        self.stage(
            "Retriever", {"observations": ["fixture"], "source_paths": ["fixture"]}
        )
        self.stage("Planner", {"design_description": "fixture"})
        with self.assertRaises(AssertionError):
            self.stage("Stylist", {"x": 1})

    def test_changed_source(self):
        (self.job / "source.txt").write_text("tamper")
        with self.assertRaises(ValueError):
            n.next_task(self.job)

    def test_prompt_binding(self):
        self.preimage()
        im = self.root / "fixture.png"
        Image.new("RGB", (2, 2)).save(im)
        with self.assertRaises(AssertionError):
            n.record(self.job, "Visualizer", im, "SYNTHETIC_TEST_NOT_LIVE", "bad")

    def test_full_synthetic_and_tamper(self):
        self.preimage()
        im = self.root / "fixture.png"
        Image.new("RGB", (2, 2)).save(im)
        n.record(
            self.job,
            "Visualizer",
            im,
            "SYNTHETIC_TEST_NOT_LIVE",
            n.next_task(self.job)["prompt_sha256"],
        )
        self.stage(
            "Critic",
            {
                "image_sha256": n.sha(im.read_bytes()),
                "verdict": "revise",
                "observed_scientific_fidelity": "synthetic",
                "aesthetics": "synthetic",
                "issues": ["fixture"],
            },
        )
        self.assertEqual(n.next_task(self.job)["status"], "REVISION_REQUIRED")
        (self.job / "image.png").write_bytes(b"tamper")
        with self.assertRaises(AssertionError):
            n.next_task(self.job)

    def test_review_wrong_image(self):
        self.preimage()
        im = self.root / "fixture.png"
        Image.new("RGB", (2, 2)).save(im)
        n.record(
            self.job,
            "Visualizer",
            im,
            "SYNTHETIC_TEST_NOT_LIVE",
            n.next_task(self.job)["prompt_sha256"],
        )
        with self.assertRaises(AssertionError):
            self.stage("Critic", {"image_sha256": "bad", "verdict": "accept"})


if __name__ == "__main__":
    unittest.main()
