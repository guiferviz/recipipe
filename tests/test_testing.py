
import pandas as pd

from unittest import TestCase

from .context import recipipe as r


def create_df_cat():
    return pd.DataFrame({
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


class DataSetTest(TestCase):

    def test_error_void_recipipe(self):
        """An error should be throw when you try to fit an empty recipipe. """
        df = create_df_all()
        t = r.recipipe() + r.scale("number") + r.onehot(["color"])
        t.fit(df)
        res = t.transform(df)
        self.assertEqual(len(res.columns), 4)
