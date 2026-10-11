"""PaperBanana's public figure direction must remain aligned with RFS."""
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class JournalStyleContractTests(unittest.TestCase):
    def test_standard_and_legacy_skill_are_identical(self):
        self.assertEqual((ROOT / "skill/SKILL.md").read_bytes(),
                         (ROOT / "skills/paperbanana/SKILL.md").read_bytes())

    def test_figure_style_is_standalone_and_scientifically_bounded(self):
        content=(ROOT / "skills/paperbanana/SKILL.md").read_text(encoding="utf-8")
        self.assertIn("three independent outputs",content)
        self.assertIn("no large figure title",content)
        self.assertIn("journal-mechanism-style.md",content)
        self.assertIn("FigureSpec",content)

    def test_engine_does_not_promise_unverified_multilabel_equations(self):
        ref=(ROOT / "skills/paperbanana/references/exchange.md").read_text(encoding="utf-8")
        self.assertIn("multiclass Dirichlet",ref)
        self.assertIn("multi-label",ref)

if __name__=="__main__":
    unittest.main()
