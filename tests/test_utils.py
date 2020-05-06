
from unittest import TestCase

import pandas as pd

from recipipe.utils import flatten_list
from recipipe.utils import fit_columns
from recipipe.utils import get_keys_eq_value
from tests.fixtures import create_df_3dtypes


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

        cols = fit_columns(create_df_3dtypes(), ["c*"])
        self.assertListEqual(cols, ["c1", "c2"])

    def test_fit_columns_dtype(self):

        cols = fit_columns(create_df_3dtypes(), dtype=int)
        self.assertListEqual(cols, ["c1", "t1"])

    def test_fit_columns_cols_dtype(self):

        cols = fit_columns(create_df_3dtypes(), ["c*"], int)
        self.assertListEqual(cols, ["c1"])

    def test_fit_columns_no_match(self):

        with self.assertRaises(ValueError):
            cols = fit_columns(create_df_3dtypes(), ["r*"])

    def test_fit_columns_no_match_no_error(self):

        cols = fit_columns(create_df_3dtypes(), ["r*"], raise_error=False)
        self.assertListEqual(cols, [])

    def test_fit_columns_no_cols_dtype(self):

        cols = fit_columns(create_df_3dtypes())
        self.assertListEqual(cols, ["c1", "c2", "t1"])

    def test_fit_columns_no_df_cols_no_cols_dtype(self):

        cols = fit_columns(pd.DataFrame())
        self.assertListEqual(cols, [])

    def test_fit_columns_no_dtype_in_df(self):

        cols = fit_columns(create_df_3dtypes(), dtype=float)
        self.assertListEqual(cols, [])

    def test_fit_columns_duplicates(self):

        cols = fit_columns(create_df_3dtypes(), cols=["c*", "c1"],
                drop_duplicates=False)
        self.assertListEqual(cols, ["c1", "c2", "c1"])

    def test_fit_columns_duplicates_drop(self):

        cols = fit_columns(create_df_3dtypes(), cols=["c*", "c1"])
        self.assertListEqual(cols, ["c1", "c2"])

    def test_fit_columns_keep_order(self):
        """Test that the columns are not alphabetically ordered. """

        df = pd.DataFrame({"c2": [], "c1": []})
        cols = fit_columns(df, ["c*"])
        self.assertListEqual(cols, ["c2", "c1"])

    def test_get_keys(self):

        d = {"a": "b", "c": "c"}
        l = get_keys_eq_value(d)
        self.assertListEqual(l, ["c"])

    def test_get_keys_empty_dict(self):

        d = {}
        l = get_keys_eq_value(d)
        self.assertListEqual(l, [])

    def test_get_keys_empty_result(self):

        d = {"ash": "pikachu"}
        l = get_keys_eq_value(d)
        self.assertListEqual(l, [])

