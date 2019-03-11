
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class Recipipe(object):

    def __init__(self):
        self.pipeline = None
        self.steps = []
        self.idx = 0

    def __add__(self, transformer):
        name = transformer.name if transformer.name is not None \
            else "step{:02d}".format(self.idx)
        self.steps.append([
            name,
            transformer
        ])
        self.idx += 1
        return self

    def _create_pipeline(self):
        return Pipeline(self.steps)

    def get_step_dict(self):
        d = {}
        for [name, o] in self.steps:
            d[name] = o
        return d

    def get_pipeline(self):
        return self.pipeline

    def fit(self, df):
        self.pipeline = self._create_pipeline()
        self.pipeline.fit(df)

    def transform(self, df):
        return self.pipeline.transform(df)


class RecipipeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, cols, name=None):
        self._cols = cols
        self._cols_fitted = None
        self.name = name

    def fit_cols(self, df):
        if type(self._cols) == list:
            self._cols_fitted = self._cols
        else:
            self._cols_fitted = df.select_dtypes(self._cols).columns
        
    def get_cols(self):
        if self._cols_fitted is None:
            raise Exception("Columns not fitted, call fit_cols before")
        return self._cols_fitted

    def fit(self, df):
        self.fit_cols(df)


class ColumnTransformer(RecipipeTransformer):

    def __init__(self, cols):
        super().__init__(cols)

    def fit(self, df, y=None):
        super().fit(df)
        self.fit_cols(df)
        for i in self.get_cols():
            self._fit_column(df, i)
        print("end fit!")
        return self

    def transform(self, df):
        for i in self.get_cols():
            df[i] = self._transform_column(df, i)
        return df


class ColumnsTransformer(RecipipeTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, df, y=None):
        super().fit(df)
        c = self.get_cols()
        self._fit_columns(df, c)
        return self

    def transform(self, df):
        c = self.get_cols()
        df[c] = self._transform_columns(df, c)
        return df


class SScaler(ColumnsTransformer):

    def __init__(self, cols, name=None):
        super().__init__(cols, name=name)
        self.ss = StandardScaler()

    def _fit_columns(self, df, c):
        self.ss.fit(df[c].values)

    def _transform_columns(self, df, c):
        return self.ss.transform(df[c].values)


class Scaler(ColumnTransformer):

    def __init__(self, cols="number", **kwargs):
        super().__init__(cols, **kwargs)
        self.means = {}
        self.stds = {}

    def _fit_column(self, df, c):
        self.means[c] = df[c].mean()
        self.stds[c] = df[c].std()

    def _transform_column(self, df, c):
        return (df[c] - self.means[c]) / self.stds[c]


class SklearnCreator(object):
    def __init__(self, sklearn_transformer):
        self.st = sklearn_transformer

    def __call__(self, *args, **kwargs):
        return SklearnWrapper(self.st, *args, **kwargs)


class SklearnWrapper(RecipipeTransformer):

    def __init__(self, sklearn_transformer, cols):
        super().__init__(cols)
        self.sklearn_transformer = sklearn_transformer
        self.new_columns = None

    def fit(self, df, y=None):
        super().fit(df)
        c = self.get_cols()
        self.sklearn_transformer.fit(df[c].values)
        if hasattr(self.sklearn_transformer, "get_feature_names"):
            self.new_cols = self.sklearn_transformer.get_feature_names(c)
        else:
            self.new_cols = ["{}_{}".format("_".join(c), i) for i in range(output.shape[1])]
        return self

    def transform(self, df):
        c = self.get_cols()
        output = self.sklearn_transformer.transform(df[c].values)
        df_new = pd.DataFrame(output, columns=self.new_cols)
        df_others = df.drop(columns=c)
        return df_others.join(df_new)
