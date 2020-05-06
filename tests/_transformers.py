
import numpy as np

import pandas as pd

import unittest

from tests.fixtures import create_df_all
from tests.fixtures import create_df_cat
from tests.fixtures import create_df_cat2

import recipipe as r


class OneHotTest(unittest.TestCase):
    """OneHot transformer test suite. """

    def test_onehot_columns_names(self):
        """Check the name of the columns after applying onehot encoding.

        We are not taking into account the order of the columns for
        this test.
        """

        df1 = create_df_cat()
        t = r.recipipe() + r.onehot()
        df2 = t.fit_transform(df1)
        expected = ["color='red'", "color='blue'"]
        self.assertCountEqual(df2.columns, expected)

    def test_onehot_columns_names_underscore(self):
        """Check columns names when "_" is present in the column name.

        We are not taking into account the order of the columns for
        this test.
        """

        df1 = create_df_cat()
        df1.columns = ["my_color"]
        t = r.recipipe() + r.onehot()
        df2 = t.fit_transform(df1)
        expected = ["my_color='red'", "my_color='blue'"]
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
            "color='red'": [1., 0, 1],
            "color='blue'": [0., 1, 0]
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
            "gender='female'": [1., 0, 0],
            "gender='male'": [0., 1, 1],
            "color='blue'": [0., 1, 0],
            "color='red'": [1., 0, 1]
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
            "color='blue'": [0., 1, 0],
            "color='red'": [1., 0, 1]
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
            "gender='female'": [1., 0, 0],
            "gender='male'": [0., 1, 1],
            "color": ["red", "blue", "red"],
        })
        self.assertTrue(expected.equals(df2))

    def test_onehot_two_columns_two_steps(self):
        """Apply onehot to two columns, once at a time. """

        df1 = create_df_cat2()
        t = r.recipipe() + r.onehot(["color"]) + r.onehot(["gender"])
        df2 = t.fit_transform(df1)
        expected = pd.DataFrame({
            "gender='female'": [1., 0, 0],
            "gender='male'": [0., 1, 1],
            "color='blue'": [0., 1, 0],
            "color='red'": [1., 0, 1]
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
            "gender='female'": [1., 0, 0],
            "gender='male'": [0., 1, 1],
            "color='blue'": [0., 1, 0],
            "color='red'": [1., 0, 1]
        })
        self.assertTrue(expected.equals(df2))

    def test_keep_cols(self):
        """Keep transformed column. """

        df1 = create_df_cat()
        t = r.recipipe() + r.onehot(keep_original=True)
        df2 = t.fit_transform(df1)
        expected = pd.DataFrame({
            "color": ["red", "blue", "red"],
            "color='blue'": [0., 1, 0],
            "color='red'": [1., 0, 1]
        })
        self.assertTrue(expected.equals(df2))

    def test_variable_args(self):
        """Check that onehot allows the use of variable-length params. """

        df1 = create_df_cat2()
        t = r.recipipe() + r.onehot("color", "gender")
        df2 = t.fit_transform(df1)
        expected = ["gender='female'", "gender='male'",
                    "color='blue'", "color='red'"]
        columns = list(df2.columns)
        self.assertEqual(columns, expected)


class SelectTest(unittest.TestCase):
    """Select transformer tests. """

    def _create_fit_transform(self, *args, **kwargs):
        """Create an apply a 'select' on the sample dataframe.

        Args:
            *args: Passed to the 'select'.
            **kwargs: Passed to the 'select'.

        Returns:
            The sample dataframe transformed using a recipipe
            that only contains a 'select'.
        """

        df = create_df_all()
        print(df.columns)
        t = r.recipipe() + r.select(*args, **kwargs)
        return t.fit_transform(df)

    def test_select_one_column(self):
        """Check select works with one column. """

        df = self._create_fit_transform("color")
        expected = ["color"]
        columns = list(df.columns)
        self.assertEqual(columns, expected)

    def test_select_two_columns(self):
        """Check that select works with two columns.

        The order of the returned columns should be the same.
        """

        df = self._create_fit_transform("color", "amount")
        expected = ["color", "amount"]
        columns = list(df.columns)
        self.assertEqual(columns, expected)


class DropTest(unittest.TestCase):

    def _create_fit_transform(self, *args, **kwargs):
        df = create_df_all()
        t = r.recipipe() + r.drop(*args, **kwargs)
        return t.fit_transform(df)

    def test_select_one_column(self):
        """Test if drop can remove one column.

        We do not need more test for drop because it uses the
        same method of fitting column names as all recipipe
        transforms.
        In this test we also check that the order of the remaining
        columns is the same after droping.
        """
        df = self._create_fit_transform("color")
        expected = ["price", "amount"]
        columns = list(df.columns)
        self.assertEqual(columns, expected)


class TestCategoryEncoder(unittest.TestCase):

    def test__fit_column(self):
        df = create_df_cat()
        t = r.category()
        t._fit_column(df, "color")
        cat = list(t.categories["color"])
        self.assertListEqual(cat, ["blue", "red"])

    def test__transform_column(self):
        df = create_df_cat()
        t = r.category()
        t.categories = {"color": pd.Index(["blue", "red"])}
        s = t._transform_column(df, "color")
        self.assertListEqual(list(s), [1, 0, 1])


class TestQueryTransformer(unittest.TestCase):

    def test_transform(self):
        df = create_df_all()
        t = r.query("color == 'red'")
        df_out = t.fit_transform(df)
        expected = pd.DataFrame({
            "color": ["red", "red"],
            "price": [1.5, 3.5],
            "amount": [1, 3],
            "index": [0, 2]
        })
        expected.set_index("index", inplace=True)
        self.assertTrue(expected.equals(df_out))


class TestReplaceTransformer(unittest.TestCase):
    def test__transform_columns_text(self):
        df_in = pd.DataFrame({
            "Vowels": ["a", "e", None, "o", "u"]
        })
        t = r.replace(values={None: "i"})
        df_out = t.fit_transform(df_in)
        expected = pd.DataFrame({
            "Vowels": ["a", "e", "i", "o", "u"]
        })
        self.assertTrue(expected.equals(df_out))


class TestGroupByTransformer(unittest.TestCase):

    def test_fit_transform(self):
        # TODO: This is not an unit test...
        df_in = pd.DataFrame({
            "color": ["red", "red", "red", "blue", "blue", "blue"],
            "other": [1, 2, 3, 4, 5, 6],
            "amount": [5, 6, 7, 1, 2, 3],
            "index": [3, 4, 5, 0, 1, 2]
        })
        # Set an unordered index to check the correct order of the output.
        df_in.set_index("index", inplace=True)
        t = r.groupby("color", r.scale("amount"))
        df_out = t.fit_transform(df_in)
        norm = 1 / np.std([1, 2, 3])
        expected = pd.DataFrame({
            "color": ["red", "red", "red", "blue", "blue", "blue"],
            "other": [1, 2, 3, 4, 5, 6],
            "amount": [-norm, 0, norm, -norm, 0, norm],
            "index": [3, 4, 5, 0, 1, 2]
        })
        expected.set_index("index", inplace=True)
        self.assertTrue(expected.equals(df_out))

