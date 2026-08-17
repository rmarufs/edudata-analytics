from io import BytesIO
import logging
import unittest
import warnings

import pandas as pd

logging.getLogger("streamlit").setLevel(logging.ERROR)

from Utils.automatic_data_processing import fill_na, standard_auto_cleaning
from Utils import upload_utils


class NamedBytesIO(BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.errors = []

    def error(self, message):
        self.errors.append(message)


class DataProcessingTests(unittest.TestCase):
    def setUp(self):
        self.original_st = upload_utils.st
        upload_utils.st = FakeStreamlit()

    def tearDown(self):
        upload_utils.st = self.original_st

    def test_load_data_preserves_text_commas_and_converts_decimal_commas(self):
        file_obj = NamedBytesIO(
            b'name,score\n"hello, world","1,5"\n"plain text","2,5"\n',
            "sample.csv",
        )

        df = upload_utils.load_data(file_obj)

        self.assertEqual(df.loc[0, "name"], "hello, world")
        self.assertEqual(df.loc[1, "name"], "plain text")
        self.assertAlmostEqual(float(df.loc[0, "score"]), 1.5)
        self.assertAlmostEqual(float(df.loc[1, "score"]), 2.5)

    def test_fill_na_methods_do_not_emit_pandas_future_warnings(self):
        df = pd.DataFrame(
            {
                "numeric": [1.0, None, 3.0],
                "category": ["a", None, "a"],
                "constant": [None, "x", None],
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FutureWarning)
            result = fill_na(df, ["numeric"], "median")
            result = fill_na(result, ["category"], "mode")
            result = fill_na(result, ["constant"], "unknown")

        self.assertEqual([w for w in caught if issubclass(w.category, FutureWarning)], [])
        self.assertEqual(result["numeric"].isna().sum(), 0)
        self.assertEqual(result["category"].isna().sum(), 0)
        self.assertEqual(result["constant"].isna().sum(), 0)

    def test_standard_auto_cleaning_removes_high_missing_columns(self):
        df = pd.DataFrame(
            {
                "mostly_empty": [None, None, None, "value"],
                "target": [1, 0, 1, 0],
                "feature": [10, 20, 30, 40],
            }
        )

        cleaned, log = standard_auto_cleaning(df)

        self.assertNotIn("mostly_empty", cleaned.columns)
        self.assertTrue(any(item["column"] == "mostly_empty" for item in log))


if __name__ == "__main__":
    unittest.main()
