"""PaperBanana discovery metadata and competing-skill intent fixtures.

These checks validate an agent-facing contract, not an LLM's measured invocation
rate. Both historical and standard SKILL.md paths remain byte-identical.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "skills" / "paperbanana" / "SKILL.md"
LEGACY = ROOT / "skill" / "SKILL.md"
UI = ROOT / "skills" / "paperbanana" / "agents" / "openai.yaml"
CORPUS = ROOT / "tests" / "agent_selection_cases.json"


class AgentSelectionContractTests(unittest.TestCase):
    def test_single_canonical_skill_root_and_legacy_content_mirror(self):
        self.assertEqual(STANDARD.read_bytes(), LEGACY.read_bytes())
        content = STANDARD.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("---\n"))
        header = content.split("\n---\n", 1)[0]
        self.assertIn("name: paperbanana\n", header)
        m = re.search(r"(?ms)^description: >-\n((?:[ ]{2}[^\n]*\n)+)", header + "\n")
        self.assertIsNotNone(m, header)
        desc = " ".join(x.strip() for x in m.group(1).splitlines())
        self.assertLess(len(desc), 800)
        for term in ("explicitly asks", "delegates", "figurespec", "do not use", "tikz", "plotter"):
            self.assertIn(term, desc.lower())
        # The legacy path is a compatibility entrypoint, not an independently
        # installable Agent Skills root (its parent directory is named 'skill').

    def test_ui_default_prompt_and_invocation_policy_remain_compatible(self):
        ui = UI.read_text(encoding="utf-8")
        self.assertIn("$paperbanana", ui)
        self.assertIn("allow_implicit_invocation: true", ui)
        self.assertIn("five-agent", ui.lower())
        self.assertIn("description", STANDARD.read_text().lower())

    def test_competing_skill_fixture_covers_positive_and_near_miss(self):
        source = json.loads(CORPUS.read_text(encoding="utf-8"))
        cases = source["cases"]
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        counts = {skill: sum(c["intended_skill"] == skill for c in cases)
                  for skill in ("paperbanana", "research-figure", "none")}
        self.assertGreaterEqual(counts["paperbanana"], 5)
        self.assertGreaterEqual(counts["research-figure"], 5)
        self.assertGreaterEqual(counts["none"], 4)
        self.assertTrue(all(c["prompt"].strip() and c["lang"] in ("zh", "en")
                            and c["intended_skill"] in counts for c in cases))
        self.assertTrue(any("$paperbanana" in c["prompt"] and
                            c["intended_skill"] == "paperbanana" for c in cases))


if __name__ == "__main__":
    unittest.main()
