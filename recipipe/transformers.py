
import abc
import fnmatch

import numpy as np

import pandas as pd

import sklearn
import sklearn.impute
from sklearn.base import TransformerMixin, clone

from recipipe.utils import default_params
from recipipe.core import RecipipeTransformer


class SelectTransformer(RecipipeTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, df):
        """Select the fitted columns. """

        cols = self.get_cols()
        return df[cols]

    def inverse_transform(self, df):
        return self.transform(df)


class DropTransformer(RecipipeTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, df):
        """Drop the fitted columns. """

        cols = self.get_cols()
        return df.drop(cols, axis=1)


class ColumnTransformer(RecipipeTransformer):

    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self, df, y=None):
        for i in self.get_cols():
            self._fit_column(df, i)
        return self

    @abc.abstractmethod
    def _fit_column(self, df, column_name):
        pass

    def _transform(self, df_in):
        df_out = pd.DataFrame(index=df_in.index)
        for i in self.get_cols():
            df_out[i] = self._transform_column(df_in, i)
        return df_out

    @abc.abstractmethod
    def _transform_column(self, df, column_name):
        pass

    def _inverse_transform_column(self, df, column_name):
        pass

    def _inverse_transform(self, df_in):
        df_out = pd.DataFrame(index=df_in.index)
        for i in self.get_cols():
            df_out[i] = self._inverse_transform_column(df_in, i)
        return df_out


class ColumnsTransformer(RecipipeTransformer):

    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self, df, y=None):
        c = self.get_cols()
        self._fit_columns(df, c)
        return self

    @abc.abstractmethod
    def _fit_columns(self, df, column_name):
        pass

    def _transform(self, df_in):
        c = self.get_cols()
        np_out = self._transform_columns(df_in, c)
        df_out = pd.DataFrame(np_out, columns=c, index=df_in.index)
        return df_out

    @abc.abstractmethod
    def _transform_columns(self, df, columns_name):
        pass


class CategoryEncoder(ColumnTransformer):

    def __init__(self, *args, error_unknown=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.categories = {}
        self.error_unknown = error_unknown

    def _fit_column(self, df, column_name):
        cat = df[column_name].astype("category").cat.categories
        # Save category values for the transformation phase.
        self.categories[column_name] = cat

    def _transform_column(self, df, col):
        encoded = pd.Categorical(df[col], categories=self.categories[col])
        # Raise exception if unknown values.
        if self.error_unknown and encoded.isna().values.any():
            raise ValueError(f"The column {col} has unknown categories")
        # Fill unknown.
        #encoded.fillna(-1)
        return encoded.codes

    def _inverse_transform_column(self, df, column_name):
        return self.categories[column_name][df[column_name].values]


class PandasScaler(ColumnTransformer):
    """Standard scaler implemented with Pandas operations. """

    def __init__(self, *args, factor=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = {}
        self.stds = {}
        self.factor = factor

    def _fit_column(self, df, c):
        self.means[c] = df[c].mean()
        self.stds[c] = df[c].std()

    def _transform_column(self, df, c):
        return (df[c] - self.means[c]) / self.stds[c] * self.factor

    def _inverse_transform_column(self, df, c):
        return (df[c] * self.stds[c] / self.factor) + self.means[c]


class SklearnCreator(object):
    """Utility class to generate SKLearn wrappers.

    Use this class to reuse any existing SKLearn transformer in your
    recipipe pipelines.

    Args:
        sklearn_transformer (sklearn.TransformerMixin): Any instance of an
            SKLearn transformer you want to incorporate in the recipipes
            pipelines.

    Example::

        # Create a onehot encoder using the SKLearn OneHotEncoder class.
        onehot = SklearnCreator(OneHotEncoder(sparse=False))
        # Now you can use the onehot variable as a transformer in a recipipe.
        recipipe() + onehot(dtype="string")
    """

    def __init__(self, sklearn_transformer, **kwargs):
        self.trans = sklearn_transformer
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        """Instantiate a SklearnWrapper using a copy of the given transformer.

        It's important to make this copy to avoid fitting several times the
        same transformer.
        """
        kwargs = default_params(kwargs, **self.kwargs)
        return SklearnWrapper(clone(self.trans), *args, **kwargs)


class SklearnWrapper(RecipipeTransformer):

    def __init__(self, sklearn_transformer, *args,
                 separator="=", quote=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.sklearn_transformer = sklearn_transformer
        self.separator = separator
        self.quote = quote
        self.new_cols_list = None
        self.new_cols_hierarchy = None

    def _fit(self, df, y=None):
        c = self.get_cols()
        self.sklearn_transformer.fit(df[c].values)
        self._set_output_names_list(df, len(c))
        return self

    def _transform(self, df):
        c = self.get_cols()
        output = self.sklearn_transformer.transform(df[c].values)
        df_new = pd.DataFrame(output, columns=self.new_cols_list, index=df.index)
        return df_new

    def _set_output_names_list(self, df, output_size):
        """Set new_cols_list and new_cols_hierarchy. """

        c = self.get_cols()

        # Check if sklearn object has features index.
        # This is common in transformers like MissingIndicator with param
        # features="missing-only", that is, transformers that do not return
        # all the input features we pass to them.
        if hasattr(self.sklearn_transformer, "features_"):
            c = [c[i] for i in self.sklearn_transformer.features_]
            output_size = len(c)

        if hasattr(self.sklearn_transformer, "get_feature_names"):
            # c_aux instead of c to avoid splitting column names with "_".
            c_aux = [str(i) for i in range(len(c))]
            new_cols = self.sklearn_transformer.get_feature_names(c_aux)
            new_cols = [i.split("_", 1) for i in new_cols]
            new_cols = [(c[int(i[0])], i[1].replace("'", "\\'")) for i in new_cols]
            new_cols_hierarchy = {i[0]: [] for i in new_cols}
            str_format = "{}{}'{}'" if self.quote else "{}{}{}"
            for k, v in new_cols:
                new_cols_hierarchy[k].append(v)
            self.new_cols_list = [str_format.format(i[0], self.separator, i[1]) for i in new_cols]
            for i in new_cols_hierarchy:
                new_cols_hierarchy[i] = tuple([str_format.format(i, self.separator, j) for j in new_cols_hierarchy[i]])
            self.new_cols_hierarchy = new_cols_hierarchy
        elif len(c) == output_size:
            self.new_cols_hierarchy = super().get_column_mapping()
            # Recreate dict just in case the number of output columns
            # is not the same as the number of input columns.
            self.new_cols_hierarchy = {i: self.new_cols_hierarchy[i] for i in c}
            self.new_cols_list = [self.new_cols_hierarchy[i] for i in c]
        else:
            name = "_".join(c)
            self.new_cols_list = [f"{name}_{i}" for i in range(output_size)]
            self.new_cols_hierarchy = {tuple(c): tuple(self.new_cols_list)}

    def get_column_mapping(self):
        return self.new_cols_hierarchy

    def _inverse_transform(self, df_in):
        c = self.get_cols()
        np_out = self.sklearn_transformer.inverse_transform(df_in[c].values)
        df_out = pd.DataFrame(np_out, columns=c, index=df_in.index)
        return df_out


class SklearnWrapperByColumn(ColumnsTransformer):

    def __init__(self, sklearn_transformer, cols=None,
                 keep_cols=False, separator="=", quote=True):
        super().__init__(cols)
        self.original_transformer = sklearn_transformer
        self.new_cols = None
        self.transformers = {}

    def _fit_column(self, df, c):
        t = clone(self.original_transformer)
        t.fit(df[c].values)
        if hasattr(t, "get_feature_names"):
            nc = [i[1:] for i in t.get_feature_names("")]
            str_format = "{}{}'{}'" if self.quote else "{}{}{}"
            nc = [str_format.format(c, self.separator, i) for i in nc]
            self.new_cols = nc
        self.transformers[c] = t
        return self

    def _transform_column(self, df, c):
        output = self.transformers[c].transform(df[c].values)
        df_new = pd.DataFrame(output, columns=self.new_cols)
        df_others = df if self.keep_cols else df.drop(columns=c)
        return df_others.join(df_new)

    def _columns_in_order(self, cols, new_cols):
        pass


class MissingIndicatorCreator(object):
    """Helper class for creating missing indicator transformers. """

    def __init__(self):
        self.kwargs = dict(col_format="INDICATOR({})")

    def __call__(self, *args, missing_values=np.nan, features="missing-only",
                 sparse="auto", error_on_new=True, **kwargs):
        """Create an SKLearn MissingIndicator using an SKLearnWrapper.

        Read the `sklearn.impute.MissingIndicator` documentation to get more
        information about the parameters.

        Returns:
            SklearnWrapper transformer with MissingIndicator as sklearn
            transformer.
        """
        kwargs = default_params(kwargs, **self.kwargs)
        mi = sklearn.impute.MissingIndicator(missing_values=missing_values,
                                             features=features,
                                             sparse=sparse,
                                             error_on_new=error_on_new)
        return SklearnWrapper(mi, *args, **kwargs)


class SimpleImputerCreator(object):
    """Helper class for creating simple imputer transformers. """

    def __call__(self, *args, missing_values=np.nan, strategy="mean",
                 fill_value=None, verbose=0, copy=True, **kwargs):
        """Create an SKLearn SimpleImputer using an SKLearnWrapper.

        Read the `sklearn.impute.SimpleImputer` documentation to get more
        information about the parameters.

        Returns:
            SklearnWrapper transformer with SimpleImputer as sklearn
            transformer.
        """
        if fill_value is not None:
            strategy = "constant"
        mi = sklearn.impute.SimpleImputer(missing_values=missing_values,
                                          strategy=strategy,
                                          fill_value=fill_value,
                                          verbose=verbose,
                                          copy=copy)
        return SklearnWrapper(mi, *args, **kwargs)


class ReplaceTransformer(ColumnsTransformer):

    def __init__(self, *args, values=None, **kwargs):
        assert values is not None
        super().__init__(*args, **kwargs)
        self.values = values

    def _transform_columns(self, df, columns_name):
        return df[columns_name].replace(self.values)


class QueryTransformer(RecipipeTransformer):

    def __init__(self, query, **kwargs):
        super().__init__(**kwargs)
        self.query = query

    def transform(self, df_in):
        return df_in.query(self.query)


class GroupByTransformer(RecipipeTransformer):
    """Apply a transformer on each group.

    Example::

        # Normalize by group using the SKLearn StandardScaler.
        scaler = SklearnCreator(StandardScaler())
        Recipipe() + GroupByTransformer("group_column", scaler("num_column"))
    """

    def __init__(self, groupby, transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.groupby = groupby
        self.transformer = transformer
        self.transformers = {}

    def fit(self, df, y=None):
        groups = df.groupby(self.groupby)
        for name, group in groups:
            self.fit_group(name, group)
        return self

    def transform(self, df_in):
        groups = df_in.groupby(self.groupby)
        dfs = []
        for name, group in groups:
            dfs.append(self.transform_group(name, group))
        df_out = pd.concat(dfs, axis=0)
        return df_out.loc[df_in.index, :]

    def fit_group(self, name, df):
        import copy
        t = copy.deepcopy(self.transformer)
        t.fit(df)
        self.transformers[name] = t

    def transform_group(self, name, df):
        return self.transformers[name].transform(df)

    def inverse_transform_group(self, name, df):
        return self.transformers[name].inverse_transform(df)

    def inverse_transform(self, df_in):
        groups = df_in.groupby(self.groupby)
        dfs = []
        for name, group in groups:
            dfs.append(self.inverse_transform_group(name, group))
        df_out = pd.concat(dfs, axis=0)
        return df_out.loc[df_in.index, :]

