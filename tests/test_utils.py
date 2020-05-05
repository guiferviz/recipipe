
from unittest import TestCase

import pandas as pd

from recipipe.utils import flatten_list
from recipipe.utils import fit_columns


DF = pd.DataFrame({"c1": [1], "c2": ["hi"], "t1": [1]})


class UtilsTest(TestCase):

    def test_flatten_one_level(self):
        """One level of nested iterators. """

        a = ["c1", ["c2"], set(["c3"]), ("c4", "c5")]
        b = ["c1", "c2", "c3", "c4", "c5"]
        self.assertEqual(flatten_list(a), b)

    def test_flatten_multi_level(self):
        """More than one level of nested iterators, testing recursivity. """

        a = ["c1", ["c2", (["c3", ("c4", [[set(["c5"])], "c6"])])]]
        b = ["c1", "c2", "c3", "c4", "c5", "c6"]
        self.assertEqual(flatten_list(a), b)

    def test_fit_columns_cols(self):

        cols = fit_columns(DF, ["c*"])
        self.assertListEqual(cols, ["c1", "c2"])

    def test_fit_columns_dtype(self):

        cols = fit_columns(DF, dtype=int)
        self.assertListEqual(cols, ["c1", "t1"])

    def test_fit_columns_cols_dtype(self):

        cols = fit_columns(DF, ["c*"], int)
        self.assertListEqual(cols, ["c1"])

    def test_fit_columns_no_match(self):

        with self.assertRaises(ValueError):
            cols = fit_columns(DF, ["r*"])

    def test_fit_columns_no_match_no_error(self):

        cols = fit_columns(DF, ["r*"], raise_error=False)
        self.assertListEqual(cols, [])

    def test_fit_columns_no_cols_dtype(self):

        cols = fit_columns(DF)
        self.assertListEqual(cols, ["c1", "c2", "t1"])

    def test_fit_columns_no_df_cols_no_cols_dtype(self):

        cols = fit_columns(pd.DataFrame())
        self.assertListEqual(cols, [])

    def test_fit_columns_no_dtype_in_df(self):

        cols = fit_columns(DF, dtype=float)
        self.assertListEqual(cols, [])

