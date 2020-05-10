
import numpy as np

import pandas as pd

from unittest import TestCase
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import Mock

from tests.fixtures import SklearnTransformerMock
from tests.fixtures import create_df_all
from tests.fixtures import create_df_int
from tests.fixtures import create_df_cat
from tests.fixtures import create_df_cat2

import recipipe as r


class SelectTransformerTest(TestCase):

    def test_select_init(self):
        """Test super init is called in the constructor. """

        t = r.SelectTransformer("holiiii")
        self.assertListEqual(t.cols_init, ["holiiii"])

    def test_select_one_column(self):
        """Check select works with one column. """

        df = create_df_all()
        t = r.SelectTransformer()
        t.cols = ["color"]
        df = t.transform(df)
        self.assertEqual(list(df.columns), ["color"])

    def _test_select_non_existing(self, error):
        df = create_df_all()
        t = r.SelectTransformer(cols_not_found_error=error)
        t.cols = ["color", "colorize"]
        df = t.transform(df)
        self.assertEqual(list(df.columns), ["color"])

    def test_select_non_existing_error(self):
        """Select non existing column raising error. """

        with self.assertRaises(KeyError):
            self._test_select_non_existing(True)

    def test_select_non_existing_no_error(self):
        """Select non existing column without error. """

        self._test_select_non_existing(False)

    def test_inverse_transform(self):
        """Inverse transform is impossible, just identity to avoid errors. """

        t = r.SelectTransformer()
        df_in = create_df_all()
        df_out = t.inverse_transform(df_in)
        self.assertTrue(df_in.equals(df_out))


class DropTransformerTest(TestCase):

    def test_drop_init(self):
        """Test super init is called in the constructor. """

        t = r.DropTransformer("phew")
        self.assertListEqual(t.cols_init, ["phew"])

    def test_drop_one_column(self):
        """Check drop works with one column. """

        df = create_df_all()
        t = r.DropTransformer()
        t.cols = ["color"]
        df = t.transform(df)
        self.assertEqual(list(df.columns), ["price", "amount"])

    def _test_drop_non_existing(self, error):
        df = create_df_all()
        t = r.DropTransformer(cols_not_found_error=error)
        t.cols = ["color", "colorize"]
        df = t.transform(df)
        self.assertEqual(list(df.columns), ["price", "amount"])

    def test_drop_non_existing_error(self):
        """Drop non existing column raising error. """

        with self.assertRaises(KeyError):
            self._test_drop_non_existing(True)

    def test_drop_non_existing_no_error(self):
        """Drop non existing column without error. """

        self._test_drop_non_existing(False)

    def test_inverse_transform(self):
        """Inverse transform is impossible, we cannot create columns.
        
        To avoid an error in the pipeline (non-existing inverse transform
        method), check that the inverse transform returns df.
        """

        t = r.DropTransformer()
        df_in = create_df_all()
        df_out = t.inverse_transform(df_in)
        self.assertTrue(df_in.equals(df_out))


class ColumnTransformerTest(TestCase):

    def test_not_1_to_1_relationship(self):
        class C(r.ColumnTransformer):
            def _get_column_mapping(self):
                return {"color": ["color1", "color2"], "price": "price"}
        t = C("color", "price")
        with self.assertRaises(ValueError):
            t.fit(create_df_all())

    def _test_transform_column_calls(self, col_format):
        t = r.ColumnTransformer("color", "price", col_format=col_format)
        t._transform_column = MagicMock(return_value=[1,1,1])
        df = create_df_all()
        t.fit_transform(df)
        calls = [call(df, "price"), call(df, "color")]
        t._transform_column.assert_has_calls(calls, any_order=True)

    def test_transform_column_calls(self):
        self._test_transform_column_calls("{}")

    def test_transform_column_calls_format(self):
        """col_format does not change the calls made to _transform_column. """

        self._test_transform_column_calls("{}_out")

    def _test_transform_column_values(self, col_format):
        t = r.ColumnTransformer("color", "price", col_format=col_format)
        t._transform_column = MagicMock(return_value=[1,1,1])
        df = create_df_all()
        df_out = t.fit_transform(df)
        df_expected = pd.DataFrame({
            col_format.format("color"): [1,1,1],
            col_format.format("price"): [1,1,1],
            "amount": [1,2,3]})
        self.assertTrue(df_out.equals(df_expected))
        self.assertFalse(df.equals(df_expected))  # No changes on input df.

    def test_transform_column_values(self):
        self._test_transform_column_values("{}")

    def test_transform_column_values_format(self):
        """col_format does not change the calls made to _transform_column. """

        self._test_transform_column_values("{}_out")

    def _test_inverse_column_transform_calls(self, col_format):
        t = r.ColumnTransformer("color", "price", col_format=col_format)
        t._transform_column = MagicMock(return_value=[1,1,1])
        t._inverse_transform_column = MagicMock(return_value=[2,2,2])
        df = create_df_all()
        df = t.fit_transform(df)
        t.inverse_transform(df)
        calls = [call(df, col_format.format("price")),
                 call(df, col_format.format("color"))]
        t._inverse_transform_column.assert_has_calls(calls, any_order=True)

    def test_inverse_column_transform_calls(self):
        self._test_inverse_column_transform_calls("{}")

    def test_inverse_column_transform_calls_format(self):
        self._test_inverse_column_transform_calls("{}_out")

    def test_inverse_transform_column_values(self):
        t = r.ColumnTransformer("color", "price", col_format="{}_out")
        t._transform_column = MagicMock(return_value=[1,1,1])
        t._inverse_transform_column = MagicMock(return_value=[2,2,2])
        df = create_df_all()
        df_out = t.fit_transform(df)
        df_out = t.inverse_transform(df_out)
        df_expected = pd.DataFrame({
            "color": [2,2,2], "price": [2,2,2], "amount": [1,2,3]})
        self.assertTrue(df_out.equals(df_expected))
        self.assertFalse(df.equals(df_expected))  # No changes on input df.


class ColumnsTransformerTest(TestCase):

    def test_transform_columns(self):
        t = r.ColumnsTransformer("color", "price")
        df = create_df_all()
        t._transform_columns = MagicMock(return_value=df)
        t.fit_transform(df)
        # TODO: Columns order does not really matter.
        t._transform_columns.assert_called_once_with(df, ["color", "price"])

    def test_transform_columns_inverse(self):
        t = r.ColumnsTransformer("color", "price", col_format="{}_out")
        df = create_df_all()
        df_out = df[["color", "price"]].copy()
        df_out.columns = ["color", "price"]
        t._inverse_transform_columns = MagicMock(return_value=df_out)
        t.fit(df)
        df = t.transform(df)
        t.inverse_transform(df)
        # TODO: Columns order does not matter.
        params = [df, ["color_out", "price_out"]]
        t._inverse_transform_columns.assert_called_once_with(*params)


class CategoryEncoderTest(TestCase):

    def test_fit_column(self):
        df = create_df_cat()
        t = r.CategoryEncoder()
        t._fit_column(df, "color")
        cat = list(t.categories["color"])
        self.assertListEqual(cat, ["blue", "red"])

    def test_transform_column(self):
        df = create_df_cat()
        t = r.CategoryEncoder()
        t.categories = {"color": pd.Index(["blue", "red"])}
        s = t._transform_column(df, "color")
        self.assertListEqual(list(s), [1, 0, 1])

    def test_fit_transform(self):
        df = create_df_all()
        t = r.CategoryEncoder("color")
        df_out = t.fit_transform(df)
        df_expected = pd.DataFrame({
            "color": [1,0,1], "price": [1.5,2.5,3.5], "amount": [1,2,3]})
        df_expected["color"] = df_expected["color"].astype("int8")
        self.assertTrue(df_out.equals(df_expected))

    def _test_fit_transform_unknown_none(self, error_unknown):
        df = create_df_all()
        t = r.CategoryEncoder("color", error_unknown=error_unknown)
        t.fit(df)
        df_in = pd.DataFrame({
            "color": ["red","yellow"], "price": [1.5,2.5], "amount": [1,2]})
        df_out = t.transform(df_in)
        df_expected = pd.DataFrame({
            "color": [1,-1], "price": [1.5,2.5], "amount": [1,2]})
        df_expected["color"] = df_expected["color"].astype("int8")
        print(df_out)
        print(df_expected)
        self.assertTrue(df_out.equals(df_expected))

    def test_fit_transform_unknown_no_error(self):
        self._test_fit_transform_unknown_none(False)

    def test_fit_transform_unknown_error(self):
        with self.assertRaisesRegex(ValueError, ".*unknown categories.*"):
            self._test_fit_transform_unknown_none(True)

    def test_fit_transform_unknown_default(self):
        df = create_df_all()
        t = r.CategoryEncoder("color", unknown_value="UNKNOWN")
        t.fit(df)
        df_in = pd.DataFrame({
            "color": ["red","yellow"], "price": [1.5,2.5], "amount": [1,2]})
        df_out = t.transform(df_in)
        df_expected = pd.DataFrame({
            "color": [2,0], "price": [1.5,2.5], "amount": [1,2]})
        df_expected["color"] = df_expected["color"].astype("int8")
        self.assertTrue(df_out.equals(df_expected))

    def test_inverse_transform_unknown_default(self):
        df = create_df_all()
        t = r.CategoryEncoder("color", unknown_value="UNKNOWN")
        t.fit(df)
        df_in_inverse = pd.DataFrame({
            "color": [2,0], "price": [1.5,2.5], "amount": [1,2]})
        df_out = t.inverse_transform(df_in_inverse)
        df_expected = pd.DataFrame({
            "color": ["red","UNKNOWN"], "price": [1.5,2.5], "amount": [1,2]})
        self.assertTrue(df_out.equals(df_expected))


class PandasScalerTest(TestCase):

    def test_scaler(self):
        df = create_df_int()
        t = r.PandasScaler("amount")
        df_out = t.fit_transform(df)
        df_expected = pd.DataFrame({"amount": [-1.,0,1]})
        self.assertTrue(df_out.equals(df_expected))

    def test_scaler_inverse(self):
        df = create_df_int()
        t = r.PandasScaler("amount")
        df_out = t.fit_transform(df)
        df_out = t.inverse_transform(df_out)
        df_expected = pd.DataFrame({"amount": [1.,2,3]})
        self.assertTrue(df_out.equals(df_expected))


class SklearnWrapperTest(TestCase):

    def test_get_column_map_features_changes_cols(self):
        sk = MagicMock()
        sk.features_ = [1]
        t = r.SklearnWrapper(sk)
        t.cols = ["amount", "color"]
        t._get_column_mapping()
        self.assertListEqual(t.cols, ["color"])

    def test_get_column_map_features(self):
        sk = SklearnTransformerMock()
        sk.features_ = [1]
        t = r.SklearnWrapper(sk)
        t.cols = ["amount", "color"]
        d = t._get_column_mapping()
        d_expected = {"color": "color"}
        self.assertDictEqual(d, d_expected)

    def _test_get_column_map_features_name(self, col_format, expected):
        sk = SklearnTransformerMock()
        sk.get_feature_names = MagicMock(return_value=["0_blue", "0_red"])
        if col_format is not None:
            t = r.SklearnWrapper(sk, col_format=col_format)
        else:
            t = r.SklearnWrapper(sk)
        t.cols = ["color"]
        d = t._get_column_mapping()
        d_expected = {"color": tuple(expected)}
        self.assertDictEqual(d, d_expected)

    def test_get_column_map_features_name_default(self):
        expected = ["color=blue", "color=red"]
        self._test_get_column_map_features_name(None, expected)

    def test_get_column_map_features_name_format_unnamed(self):
        expected = ["color='blue'", "color='red'"]
        col_format = "{}='{}'"
        self._test_get_column_map_features_name(col_format, expected)

    def test_get_column_map_features_name_format_named(self):
        expected = ["blue(color)", "red(color)"]
        col_format = "{value}({column})"
        self._test_get_column_map_features_name(col_format, expected)

    def test_fit_transform_features_name_result(self):
        sk = SklearnTransformerMock()
        sk.get_feature_names = MagicMock(return_value=["0_blue", "0_red"])
        return_value = np.array([[1,2,3], [3,2,1]]).T
        sk.transform = MagicMock(return_value=return_value)
        t = r.SklearnWrapper(sk, "color")
        df_out = t.fit_transform(create_df_all())
        df_expected = pd.DataFrame({
            "color=blue": [1,2,3],
            "color=red": [3,2,1],
            "price": [1.5,2.5,3.5],
            "amount": [1,2,3]})
        sk.transform.assert_called_once()
        self.assertTrue(df_out.equals(df_expected))


class OneHotTransformerTest(TestCase):
    """OneHot transformer test suite. """

    def test_onehot_columns_names(self):
        """Check the name of the columns after applying onehot encoding.

        We are not taking into account the order of the columns for this test.
        """

        df1 = create_df_cat()
        t = r.recipipe() + r.onehot()
        df2 = t.fit_transform(df1)
        expected = ["color=red", "color=blue"]
        self.assertCountEqual(df2.columns, expected)

    def test_onehot_columns_names_underscore(self):
        """Check columns names when "_" is present in the column name.

        This is important because SKLearn `get_feature_names` method returns
        names with "_" as a separator.
        Not taking into account the order of the columns for this test.
        """

        df1 = create_df_cat()
        df1.columns = ["my_color"]
        t = r.recipipe() + r.onehot()
        df2 = t.fit_transform(df1)
        expected = ["my_color=red", "my_color=blue"]
        self.assertCountEqual(df2.columns, expected)

    def test_onehot_columns_values(self):
        """Check the values of a column after applying onehot encoding.

        The order of the resulting columns should be first the blue
        color and second the red color (alphabetically sorted).
        This test does not check the column names, only the values.
        """

        df1 = create_df_cat()
        t = r.recipipe() + r.onehot()
        df2 = t.fit_transform(df1)
        # Ignore the column names, we assign a new ones.
        df2.columns = ["color_blue", "color_red"]
        expected1 = pd.DataFrame({
            "color_blue": [0., 1, 0],
            "color_red": [1., 0, 1]
        })
        self.assertTrue(expected1.equals(df2))

    def test_onehot_columns_names_and_values(self):
        """Check the output dataframe after applying onehot encoding.

        For passing this test the order of the resulting columns
        does not matter.
        """

        df1 = create_df_cat()
        t = r.recipipe() + r.onehot()
        df2 = t.fit_transform(df1)
        expected = pd.DataFrame({
            "color=red": [1., 0, 1],
            "color=blue": [0., 1, 0]
        })
        self.assertTrue(expected.eq(df2).all().all())

    def test_onehot_two_columns(self):
        """Test onehot over dataframes with more than 2 category columns.
        
        The output dataframe should contain the columns in the same order.
        In this example, first all the gender onehot columns and second
        all the color columns.
        As before, the onehot columns of one column should be ordered
        alphabetically.
        """

        df1 = create_df_cat2()
        t = r.recipipe() + r.onehot()
        df2 = t.fit_transform(df1)
        expected = pd.DataFrame({
            "gender=female": [1., 0, 0],
            "gender=male": [0., 1, 1],
            "color=blue": [0., 1, 0],
            "color=red": [1., 0, 1]
        })
        self.assertTrue(expected.equals(df2))

    def test_onehot_one_of_two_columns_last(self):
        """Apply onehot only to one column, the last one on the input df.

        The non-onehotencoded column should be in the output dataframe.
        """

        df1 = create_df_cat2()
        t = r.recipipe() + r.onehot(["color"])
        df2 = t.fit_transform(df1)
        expected = pd.DataFrame({
            "gender": ["female", "male", "male"],
            "color=blue": [0., 1, 0],
            "color=red": [1., 0, 1]
        })
        self.assertTrue(expected.equals(df2))

    def test_onehot_one_of_two_columns_first(self):
        """Apply onehot only to one column, the first one on the input df.

        The non-onehotencoded column should be in the output dataframe
        and exactly in the same order.
        """

        df1 = create_df_cat2()
        # Check that the fixture df is in the expected order.
        self.assertListEqual(list(df1.columns), ["gender", "color"])

        t = r.recipipe() + r.onehot(["gender"])
        df2 = t.fit_transform(df1)
        expected = pd.DataFrame({
            "gender=female": [1., 0, 0],
            "gender=male": [0., 1, 1],
            "color": ["red", "blue", "red"],
        })
        self.assertTrue(expected.equals(df2))

    def test_onehot_two_columns_two_steps(self):
        """Apply onehot to two columns, once at a time. """

        df1 = create_df_cat2()
        t = r.recipipe() + r.onehot(["color"]) + r.onehot(["gender"])
        df2 = t.fit_transform(df1)
        expected = pd.DataFrame({
            "gender=female": [1., 0, 0],
            "gender=male": [0., 1, 1],
            "color=blue": [0., 1, 0],
            "color=red": [1., 0, 1]
        })
        self.assertTrue(expected.equals(df2))

    def test_onehot_two_columns_one_step(self):
        """Apply onehot to two columns at the same time.

        The order of the output columns is given by the
        input dataset, not by the order of the list passed
        to the one hot transformer.
        """

        df1 = create_df_cat2()
        t = r.recipipe() + r.onehot(["color", "gender"])
        df2 = t.fit_transform(df1)
        expected = pd.DataFrame({
            "gender=female": [1., 0, 0],
            "gender=male": [0., 1, 1],
            "color=blue": [0., 1, 0],
            "color=red": [1., 0, 1]
        })
        self.assertTrue(expected.equals(df2))

    def test_keep_cols(self):
        """Keep transformed column. """

        df1 = create_df_cat()
        t = r.recipipe() + r.onehot(keep_original=True)
        df2 = t.fit_transform(df1)
        expected = pd.DataFrame({
            "color": ["red", "blue", "red"],
            "color=blue": [0., 1, 0],
            "color=red": [1., 0, 1]
        })
        self.assertTrue(expected.equals(df2))

    def test_variable_args(self):
        """Check that onehot allows the use of variable-length params. """

        df1 = create_df_cat2()
        t = r.recipipe() + r.onehot("color", "gender")
        df2 = t.fit_transform(df1)
        expected = ["gender=female", "gender=male",
                    "color=blue", "color=red"]
        columns = list(df2.columns)
        self.assertEqual(columns, expected)


class QueryTransformerTest(TestCase):

    def test_transform(self):
        df = create_df_all()
        t = r.QueryTransformer("color == 'red'")
        df_out = t.fit_transform(df)
        expected = pd.DataFrame({
            "color": ["red", "red"],
            "price": [1.5, 3.5],
            "amount": [1, 3],
            "index": [0, 2]
        })
        expected.set_index("index", inplace=True)
        self.assertTrue(expected.equals(df_out))


class ReplaceTransformerTest(TestCase):

    def test__transform_columns_text(self):
        df_in = pd.DataFrame({
            "vowels": ["a", "e", None, "o", "u"]
        })
        t = r.ReplaceTransformer(values={None: "i"})
        df_out = t.fit_transform(df_in)
        expected = pd.DataFrame({
            "vowels": ["a", "e", "i", "o", "u"]
        })
        self.assertTrue(expected.equals(df_out))


class GroupByTransformerTest(TestCase):

    # TODO: This is not an unit test...
    def test_fit_transform(self):
        df_in = pd.DataFrame({
            "color": ["red", "red", "red", "blue", "blue", "blue"],
            "other": [1, 2, 3, 4, 5, 6],
            "amount": [5, 6, 7, 1, 2, 3],
            "index": [3, 4, 5, 0, 1, 2]
        })
        # Set an unordered index to check the correct order of the output.
        df_in.set_index("index", inplace=True)

        norm = 1 / np.std([1, 2, 3])
        df_expected = pd.DataFrame({
            "color": ["red", "red", "red", "blue", "blue", "blue"],
            "other": [1, 2, 3, 4, 5, 6],
            "amount": [-1., 0, 1, -1, 0, 1],
            "index": [3, 4, 5, 0, 1, 2]
        })
        df_expected.set_index("index", inplace=True)

        t = r.GroupByTransformer("color", r.PandasScaler("amount"))
        df_out = t.fit_transform(df_in)
        self.assertTrue(df_out.equals(df_expected))

