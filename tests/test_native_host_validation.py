"""Synthetic stage/repair boundary regressions; no model or network execution."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from skill import native_host as n

SPEC = Path(__file__).parent / "fixtures/native-host.figure-spec.json"


class NativeValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.job = self.root / "job"
        n.prepare(SPEC, self.job)

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, stage, data, job=None):
        artifact = self.root / (stage + ".input.json")
        artifact.write_text(json.dumps({"status": "succeeded", **data}))
        return n.record(job or self.job, stage, artifact, "SYNTHETIC_TEST_NOT_LIVE")

    def plan(self):
        self.record(
            "Retriever",
            {"observations": ["Synthetic fixture"], "source_paths": ["fixture"]},
        )
        self.record("Planner", {"design_description": "Synthetic two-node layout"})

    def style(self, **extra):
        return self.record(
            "Stylist",
            {
                "prompt": "Synthetic test diagram",
                "design_description": "Synthetic layout",
                **extra,
            },
        )

    def finish(self, verdict, job=None):
        job = job or self.job
        image = self.root / "synthetic.png"
        Image.new("RGB", (4, 4), "white").save(image)
        task = n.next_task(job)
        n.record(
            job, "Visualizer", image, "SYNTHETIC_TEST_NOT_LIVE", task["prompt_sha256"]
        )
        return self.record(
            "Critic",
            {
                "image_sha256": n.sha(image.read_bytes()),
                "verdict": verdict,
                "observed_scientific_fidelity": "Synthetic fixture only",
                "aesthetics": "Synthetic fixture only",
                "issues": [] if verdict == "accept" else ["Synthetic spacing issue"],
            },
            job,
        )

    def test_invalid_style_reference_does_not_commit(self):
        self.plan()
        with self.assertRaises((AssertionError, ValueError)):
            self.style(referenced_image_paths=["unbound.png"])
        self.assertFalse((self.job / "Stylist.json").exists())
        self.assertFalse((self.job / "Stylist.output.json").exists())
        self.assertEqual(n.next_task(self.job)["stage"], "Stylist")
        self.assertEqual(self.style()["stage"], "Visualizer")

    def test_accept_is_visual_only(self):
        self.plan()
        self.style()
        result = self.finish("accept")
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertEqual(result["scientific_truth"], "NOT_ESTABLISHED_BY_IMAGE_REVIEW")

    def test_context_tamper_rejected(self):
        ref = self.root / "context.txt"
        ref.write_text("Synthetic context")
        (self.job / "context.json").write_text(
            json.dumps(
                {"files": [{"path": str(ref), "sha256": n.sha(ref.read_bytes())}]}
            )
        )
        self.plan()
        ref.write_text("Changed context")
        with self.assertRaises((AssertionError, ValueError)):
            n.next_task(self.job)

    def test_full_repair_both_prompt_modes(self):
        self.plan()
        self.style()
        self.finish("revise")
        original = {p.name: p.read_bytes() for p in self.job.iterdir() if p.is_file()}
        _, spec, _ = n.state(self.job)
        ids = {
            entity["id"]
            for key in ("nodes", "edges", "groups", "labels", "datasets")
            for entity in spec[key]
        }
        target = sorted(ids)[0]
        edit = self.root / "edit.json"
        edit.write_text(
            json.dumps(
                {
                    "instruction": "Adjust spacing only",
                    "target_ids": [target],
                    "allowed_changes": ["spacing"],
                    "protected_ids": sorted(ids - {target}),
                }
            )
        )
        for repair_only in (False, True):
            with self.subTest(repair_only=repair_only):
                out = self.root / ("repair-only" if repair_only else "repair-append")
                task = n.repair_job(self.job, out, edit, repair_only)
                self.assertEqual(task["stage"], "Visualizer")
                self.assertEqual(
                    task["tool_arguments"]["referenced_image_paths"],
                    [str(out / "repair_reference.png")],
                )
                self.assertEqual(
                    "Synthetic test diagram" in task["tool_arguments"]["prompt"],
                    not repair_only,
                )
                self.assertEqual(self.finish("accept", out)["status"], "ACCEPTED")
        self.assertEqual(
            original,
            {p.name: p.read_bytes() for p in self.job.iterdir() if p.is_file()},
        )

    def test_optimized_python_refused(self):
        result = subprocess.run(
            [
                sys.executable,
                "-O",
                str(Path(n.__file__)),
                "next",
                "--job",
                str(self.job),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without -O", result.stderr)


if __name__ == "__main__":
    unittest.main()
