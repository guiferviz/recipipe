
import time

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from recipipe.core import RecipipeTransformer


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


class TransformerMock:
    """Mock implementing the Transformer API without inheritance. """

    def __init__(self):
        self.n_fit = 0
        self.n_transform = 0
        self.n_inverse_transform = 0

    def fit(self, X, y):
        self.n_fit += 1
        return self

    def transform(self, X):
        self.n_transform += 1
        return X

    def inverse_transform(self, X):
        self.n_inverse_transform += 1
        return X


class SklearnTransformerMock(BaseEstimator, TransformerMixin):
    """Mock implementing the SKLearn APIs with inheritance. """

    def __init__(self):
        self.n_fit = 0
        self.n_transform = 0
        self.n_inverse_transform = 0

    def fit(self, X, y):
        self.n_fit += 1
        return self

    def transform(self, X):
        self.n_transform += 1
        return X

    def inverse_transform(self, X):
        self.n_inverse_transform += 1
        return X


class RecipipeTransformerMock(RecipipeTransformer):
    """Mock implementing RecipipeTransformer API with inheritance. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_fit = 0
        self.n_transform = 0
        self.n_inverse_transform = 0

    def _fit(self, X, y):
        self.n_fit += 1
        return self

    def _transform(self, X):
        self.n_transform += 1
        return X

    def _inverse_transform(self, X):
        self.n_inverse_transform += 1
        return X

