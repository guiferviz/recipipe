
import pandas as pd

from unittest import TestCase

from .context import recipipe as r
from .fixtures import create_df_cat


class TestCategoryEncoder(TestCase):

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
