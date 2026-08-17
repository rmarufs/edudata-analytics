from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryDocsTests(unittest.TestCase):
    def test_readme_does_not_reference_missing_supabase_setup_doc(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("SUPABASE_SETUP.md", readme)
        self.assertTrue((ROOT / "supabase_schema.sql").exists())


if __name__ == "__main__":
    unittest.main()
