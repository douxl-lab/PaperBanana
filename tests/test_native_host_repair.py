"""SYNTHETIC repair-only tests; never touches production jobs or calls image tools."""

import json, unittest
from test_native_host_guards import HostFixture, n


class RepairGuards(HostFixture, unittest.TestCase):
    def edit(self):
        _, s, _ = n.state(self.job)
        ids = {
            i["id"]
            for k in ["nodes", "edges", "groups", "labels", "datasets"]
            for i in s[k]
        }
        target = sorted(ids)[0]
        p = self.root / "edit.json"
        p.write_text(
            json.dumps(
                {
                    "instruction": "SYNTHETIC test adjust label spacing only",
                    "target_ids": [target],
                    "allowed_changes": ["label spacing"],
                    "protected_ids": sorted(ids - {target}),
                }
            )
        )
        return p

    def test_repair_next_visualizer_preserves_source(self):
        self.full()
        before = {
            p.name: n.sha(p.read_bytes()) for p in self.job.iterdir() if p.is_file()
        }
        out = self.root / "repair"
        task = n.repair_job(self.job, out, self.edit())
        self.assertEqual(task["stage"], "Visualizer")
        self.assertEqual(
            task["tool_arguments"]["referenced_image_paths"],
            [str(out / "repair_reference.png")],
        )
        self.assertIn("BOUNDED REFERENCE EDIT", task["tool_arguments"]["prompt"])
        self.assertEqual(
            before,
            {p.name: n.sha(p.read_bytes()) for p in self.job.iterdir() if p.is_file()},
        )
        for role in ["Retriever", "Planner", "Stylist"]:
            self.assertEqual(
                (self.job / (role + ".json")).read_bytes(),
                (out / (role + ".json")).read_bytes(),
            )

    def test_repair_reject_nonrevise_source(self):
        self.full()
        payload = n.read_object(self.job / "Critic.output.json")
        payload["verdict"] = "accept"
        (self.job / "Critic.output.json").write_text(json.dumps(payload))
        receipt = n.read_object(self.job / "Critic.json")
        receipt["verdict"] = "accept"
        receipt["artifacts"]["Critic.output.json"] = n.sha(
            (self.job / "Critic.output.json").read_bytes()
        )
        (self.job / "Critic.json").write_text(json.dumps(receipt))
        self.assertEqual(n.next_task(self.job)["status"], "ACCEPTED")
        with self.assertRaises(AssertionError):
            n.repair_job(self.job, self.root / "repair", self.edit())
        self.assertFalse((self.root / "repair").exists())

    def test_repair_reject_changed_reference(self):
        self.full()
        out = self.root / "repair"
        n.repair_job(self.job, out, self.edit())
        (out / "repair_reference.png").write_bytes(b"SYNTHETIC tamper")
        with self.assertRaises(AssertionError):
            n.next_task(out)


if __name__ == "__main__":
    unittest.main()
