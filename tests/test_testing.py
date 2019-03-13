
import numpy as np

import pandas as pd

from unittest import TestCase

from .context import recipipe as r


def create_df_cat():
    return pd.DataFrame({
        "color": ["red", "blue", "red"]
    })

def create_df_cat2():
    return pd.DataFrame({
        "gender": ["female", "male", "male"],
        "color": ["red", "blue", "red"]
    })

def create_df_float():
    return pd.DataFrame({
        "price": [1.5, 2.5, 3.5]
    })

def create_df_int():
    return pd.DataFrame({
        "amount": [1, 2, 3]
    })

def create_df_all():
    return pd.concat([
        create_df_cat(),
        create_df_float(),
        create_df_int()
    ], axis=1)


class RecipipeTest(TestCase):

    def test_error_empty_recipipe(self):
        """An error should be throw when you try to fit an empty recipipe. """
        df = create_df_all()
        t = r.recipipe()
        with self.assertRaises(Exception):
            t.fit(df)


class SelectTest(TestCase):

    def _create_fit_transform(self, *args, **kwargs):
        df = create_df_all()
        t = r.recipipe() + r.select(*args, **kwargs)
        return t.fit_transform(df)

    def test_select_one_column(self):
        df = self._create_transform("color")
        expected = ["color"]
        columns = list(df.columns)
        self.assertEqual(columns, expected)

    def test_select_two_columns(self):
        df = self._create_fit_transform("color", "amount")
        expected = ["color", "amount"]
        columns = list(df.columns)
        self.assertEqual(columns, expected)


class DropTest(TestCase):

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


class OneHotTest(TestCase):

    def test_onehot_columns_names(self):
        """Check the name of the columns after applying onehot encoding. """
        df1 = create_df_cat()
        t = r.recipipe() + r.onehot()
        df2 = t.fit_transform(df1)
        expected = ["color_red", "color_blue"]
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
        # Ignore the column names.
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
            "color_red": [1., 0, 1],
            "color_blue": [0., 1, 0]
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
            "gender_female": [1., 0, 0],
            "gender_male": [0., 1, 1],
            "color_blue": [0., 1, 0],
            "color_red": [1., 0, 1]
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
            "color_blue": [0., 1, 0],
            "color_red": [1., 0, 1]
        })
        self.assertTrue(expected.equals(df2))

    def test_onehot_one_of_two_columns_first(self):
        """Apply onehot only to one column, the last one on the input df.

        The non-onehotencoded column should be in the output dataframe.
        """
        df1 = create_df_cat2()
        t = r.recipipe() + r.onehot(["gender"])
        df2 = t.fit_transform(df1)
        expected = pd.DataFrame({
            "gender_female": [1., 0, 0],
            "gender_male": [0., 1, 1],
            "color": ["red", "blue", "red"],
        })
        self.assertTrue(expected.equals(df2))

    def test_onehot_two_columns_two_steps(self):
        """Apply onehot to two columns, once at a time. """
        df1 = create_df_cat2()
        t = r.recipipe() + r.onehot(["color"]) + r.onehot(["gender"])
        df2 = t.fit_transform(df1)
        expected = pd.DataFrame({
            "gender_female": [1., 0, 0],
            "gender_male": [0., 1, 1],
            "color_blue": [0., 1, 0],
            "color_red": [1., 0, 1]
        })
        self.assertTrue(expected.equals(df2))
