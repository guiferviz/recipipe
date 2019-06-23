"""Onehot transformer tests. """

from unittest import TestCase

import pandas as pd

from .context import recipipe as r
from .fixtures import create_df_cat
from .fixtures import create_df_cat2


class OneHotTest(TestCase):
    """OneHot test suite. """

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
        expected = ["gender='female'", "gender='male'", "color='blue'", "color='red'"]
        columns = list(df2.columns)
        self.assertEqual(columns, expected)
