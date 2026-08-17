import subprocess
import sys
import tempfile
from pathlib import Path
import py_compile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryHealthTests(unittest.TestCase):
    def test_all_tracked_python_files_compile(self):
        result = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        python_files = [ROOT / item for item in result.stdout.splitlines()]
        self.assertTrue(python_files)

        old_prefix = sys.pycache_prefix
        with tempfile.TemporaryDirectory() as cache_dir:
            sys.pycache_prefix = cache_dir
            try:
                for path in python_files:
                    with self.subTest(path=path.relative_to(ROOT)):
                        py_compile.compile(str(path), doraise=True)
            finally:
                sys.pycache_prefix = old_prefix


if __name__ == "__main__":
    unittest.main()
