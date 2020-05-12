
import abc
import collections
import copy
import fnmatch
import inspect

import numpy as np

import pandas as pd

import sklearn
import sklearn.impute
from sklearn.base import TransformerMixin, clone

from recipipe.utils import default_params
from recipipe.utils import flatten_list
from recipipe.core import RecipipeTransformer


class SelectTransformer(RecipipeTransformer):
    """Select the fitted columns and ignore the rest of them. """

    def transform(self, df):
        """Select the fitted columns.

        Args:
            df (:obj:`pandas.DataFrame`): DataFrame to select columns from.

        Returns:
            Transformed DataFrame.
        """

        cols = self.cols
        if not self.cols_not_found_error:
            cols = [i for i in self.cols if i in df.columns]
        return df[cols]

    def inverse_transform(self, df):
        """No inverse exists for a select operation but...

        Obviously, there is no way to get back the non-selected columns, but
        it's useful to have this operation defined to avoid errors when using
        this transformer in a pipeline.
        For that reason, the inverse in defined as the identity function.

        Args:
            df (:obj:`pandas.DataFrame`): DataFrame to inverse transform.

        Returns:
            Identity function: `df` without modifications.
        """

        return df


class DropTransformer(RecipipeTransformer):
    """Drop the fitted columns and continue with the reminded ones. """

    def transform(self, df):
        """Drop the fitted columns.

        Args:
            df (:obj:`pandas.DataFrame`): DataFrame to drop columns from.

        Returns:
            Transformed DataFrame.
        """

        errors = "raise" if self.cols_not_found_error else "ignore"
        return df.drop(self.cols, axis=1, errors=errors)

    def inverse_transform(self, df):
        """No inverse exists for a drop operation but...

        Obviously, there is no way to get back the dropped columns, but
        it's useful to have this operation defined to avoid errors when using
        this transformer in a pipeline.
        For that reason, the inverse in defined as the identity function.

        Args:
            df (:obj:`pandas.DataFrame`): DataFrame to inverse transform.

        Returns:
            Identity function: `df` without modifications.
        """

        return df


class ColumnTransformer(RecipipeTransformer):
    """Apply an operation per each input column.

    This transformer only allows 1 to N relationships between input and output
    columns.
    If you want to create a column from two existing ones (like concatenate one
    column to another) this transformer is not for you.

    Note that the output number of rows of this transformer should be the same
    as in the input DataFrame. No deletes are supported here.
    """

    def fit(self, df, y=None):
        super().fit(df, y)
        if any(len(i) > 1 for i in self.col_map_1_n_inverse.values()):
            raise ValueError("Only 1 to N relationships between input and "
                             "output columns are supported by the "
                             "ColumnTransformer")
        return self

    def _fit(self, df):
        for i in self.cols:
            self._fit_column(df, i)

    def _fit_column(self, df, column_name):
        pass

    def _transform(self, df_in):
        dfs_out = []
        for i in self.cols:
            c = self.col_map_1_n[i]
            np_out = self._transform_column(df_in, i)
            df = pd.DataFrame(np_out, index=df_in.index, columns=c)
            dfs_out.append(df)
        df_out = pd.concat(dfs_out, axis=1)
        return df_out

    def _transform_column(self, df, column_name):  # pragma: no cover
        return df[column_name]

    def _inverse_transform_column(self, df, column_names):  # pragma: no cover
        """Inverse the transform on the given columns.

        `column_names` can receive more than one column name because the
        column transformer accepts 1 to N transformations.
        Note that if you are working with a 1 to 1 transformation, this
        argument is going to be a also a list (with one element, but a list).

        Args:
            df (:obj:`pandas.DataFrame`): Input DataFrame to transform.
            column_names (:obj:`list`): List of columns names to inverse
                transform.

        Returns:
            A DataFrame or a :obj:`numpy.ndarray` of one column.
            By default, it returns `df[column_names]`, so it could fail when
            used with 1 to N transformers as it will return more than one
            column.
        """

        return df[column_names]

    def _inverse_transform(self, df_in):
        df_out = pd.DataFrame(index=df_in.index, columns=self.cols)
        for i in self.cols:
            c = self.col_map_1_n[i]
            df_out[i] = self._inverse_transform_column(df_in, c)
        return df_out


class ColumnsTransformer(RecipipeTransformer):
    """Apply an operation at all the input columns at the same time.

    This class does not do that much... It's only particular useful when
    working with transformers that return a :obj:`numpy.ndarray`, like the
    ones in SKLearn. This class deals with the creation of a DataFrame so
    yo do not need to create it by yourself.

    Note that the output number of rows of this transformer should be the same
    as in the input DataFrame. No deletes are supported here.
    """

    def _fit(self, df, y=None):
        pass

    def _transform(self, df_in):
        np_out = self._transform_columns(df_in, self.cols)
        df_out = pd.DataFrame(np_out, columns=self.cols_out, index=df_in.index)
        return df_out

    def _transform_columns(self, df, column_names):
        """Transform the given columns.

        Args:
            df (:obj:`pandas.DataFrame`): Input DataFrame.
            column_names (:obj:`list`): List of columns. `df` can contain more
                columns apart from the ones in `columns_name`.

        Returns:
            A :obj:`numpy.ndarray` or :obj:`pandas.DataFrame` with the
            transformed columns.
        """

        pass

    def _inverse_transform(self, df_in):
        np_out = self._inverse_transform_columns(df_in, self.cols_out)
        df_out = pd.DataFrame(np_out, columns=self.cols, index=df_in.index)
        return df_out

    def _inverse_transform_columns(self, df, columns_name):  # pragma: no cover
        pass


class CategoryEncoder(ColumnTransformer):

    def __init__(self, *args, error_unknown=False, unknown_value=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.categories = {}
        self.error_unknown = error_unknown
        self.unknown_value = unknown_value

    def _fit_column(self, df, col):
        cat = df[col].astype("category").cat.categories
        if self.unknown_value is not None:
            cat = cat.insert(0, self.unknown_value)
        # Save category values for the transformation phase.
        self.categories[col] = cat

    def _transform_column(self, df, col):
        encoded = pd.Categorical(df[col], categories=self.categories[col])
        # Raise exception if unknown values.
        if self.error_unknown and encoded.isna().any():
            raise ValueError(f"The column {col} has unknown categories")
        # Fill unknown.
        if self.unknown_value is not None:
            encoded = encoded.fillna(self.unknown_value)
        return encoded.codes

    def _inverse_transform_column(self, df, col):
        col = col[0]
        return self.categories[col][df[col].values]


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

    def _inverse_transform_column(self, df, column_names):
        c = column_names[0]
        return (df[c] * self.stds[c] / self.factor) + self.means[c]


class SklearnCreator(object):
    """Utility class to generate SKLearn wrappers.

    Use this class to reuse any existing SKLearn transformer in your
    recipipe pipelines.

    Args:
        sk_transformer (sklearn.TransformerMixin): Any instance of an
            SKLearn transformer you want to incorporate in the recipipes
            pipelines.

    Example::

        # Create a onehot encoder using the SKLearn OneHotEncoder class.
        onehot = SklearnCreator(OneHotEncoder(sparse=False))
        # Now you can use the onehot variable as a transformer in a recipipe.
        recipipe() + onehot(dtype="string")
    """

    def __init__(self, sk_transformer, **kwargs):
        self.sk_transformer = sk_transformer
        self.kwargs = kwargs

    def __call__(self, *args, wrapper="columns", **kwargs):
        """Instantiate a SKLearn wrapper using a copy of the given transformer.

        It's important to make this copy to avoid fitting several times the
        same transformer.
        """

        wrapping_methods = ["column", "columns"]
        if wrapper not in wrapping_methods:
            raise ValueError("Wrapper method not in {wrapping_methods}")

        # SKLearn transformer params.
        signature = inspect.signature(self.sk_transformer.__init__)
        params = [i.name for i in signature.parameters.values()]
        sk_params = {}
        for i in params:
            if i in kwargs:
                sk_params[i] = kwargs[i]
                del kwargs[i]
        t = clone(self.sk_transformer)
        t.set_params(**sk_params)

        # Recipipe transformer params.
        if "recipipe_params" in kwargs:
            kwargs = default_params(kwargs, kwargs["recipipe_params"])
            del kwargs["recipipe_params"]
        kwargs = default_params(kwargs, **self.kwargs)

        if wrapper == "columns":
            return SklearnColumnsWrapper(t, *args, **kwargs)
        return SklearnColumnWrapper(t, *args, **kwargs)


class SklearnColumnWrapper(ColumnTransformer):

    def __init__(self, sk_transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_transformer = sk_transformer
        self.transformers = {}

    def _fit_column(self, df, column_name):
        t = clone(self.original_transformer)
        t.fit(df[column_name].values)
        self.transformers[column_name] = t

    def _transform_column(self, df, c):
        return self.transformers[c].transform(df[c].values)

    def _get_column_mapping(self):
        # Check if SKLearn object has features index.
        # "features_" is common in transformers like MissingIndicator with
        # the parameter features="missing-only", that is, transformers that do
        # not return all the input features we pass to them.
        cols = []
        for c, t in self.transformers.items():
            if hasattr(t, "features_"):
                if t.features_:
                    cols.append(c)
            else:
                cols.append(c)
        self.cols = cols

        col_map = super()._get_column_mapping()
        for c, t in self.transformers.items():
            if hasattr(t, "get_feature_names"):
                col_format = self.col_format
                if col_format == "{}": col_format = "{column}={value}"
                new_cols = t.get_feature_names(["0"])
                col_map[c] = []
                for i in new_cols:
                    _, value = i.split("_", 1)
                    full_name = col_format.format(c, value,
                            column=c, value=value)
                    col_map[c].append(full_name)
                if new_cols:
                    col_map[c] = tuple(col_map[c])
        return col_map


class SklearnColumnsWrapper(ColumnsTransformer):

    def __init__(self, sk_transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sk_transformer = sk_transformer

    def _fit(self, df, y=None):
        self.sk_transformer.fit(df[self.cols].values)

    def _transform_columns(self, df, cols):
        return self.sk_transformer.transform(df[cols].values)

    def _get_column_mapping(self):
        # Check if SKLearn object has features index.
        # "features_" is common in transformers like MissingIndicator with
        # the parameter features="missing-only", that is, transformers that do
        # not return all the input features we pass to them.
        if hasattr(self.sk_transformer, "features_"):
            self.cols = [self.cols[i] for i in self.sk_transformer.features_]

        if hasattr(self.sk_transformer, "get_feature_names"):
            if self.col_format == "{}": self.col_format = "{column}={value}"
            # c_aux instead of self.cols to avoid splitting column names
            # with "_",
            # This is important because SKLearn `get_feature_names` method
            # returns names with "_" as a separator.
            c_aux = list(map(str, range(len(self.cols))))
            new_cols = self.sk_transformer.get_feature_names(c_aux)
            col_map = collections.defaultdict(list)
            for i in new_cols:
                col, value = i.split("_", 1)
                col = self.cols[int(col)]
                full_name = self.col_format.format(col, value,
                        column=col, value=value)
                col_map[col].append(full_name)
            col_map = {k: tuple(v) for k, v in col_map.items()}
        else:
            col_map = super()._get_column_mapping()
        """
        else:
            name = "_".join(c)
            self.cols_out = [f"{name}_{i}" for i in range(output_size)]
            col_map = {tuple(c): tuple(self.new_cols_list)}
        """
        return col_map


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
            SklearnColumnsWrapper transformer with MissingIndicator as sklearn
            transformer.
        """
        kwargs = default_params(kwargs, **self.kwargs)
        mi = sklearn.impute.MissingIndicator(missing_values=missing_values,
                                             features=features,
                                             sparse=sparse,
                                             error_on_new=error_on_new)
        return SklearnColumnsWrapper(mi, *args, **kwargs)


class SimpleImputerCreator(object):
    """Helper class for creating simple imputer transformers. """

    def __call__(self, *args, missing_values=np.nan, strategy="mean",
                 fill_value=None, verbose=0, copy=True, **kwargs):
        """Create an SKLearn SimpleImputer using an SKLearnWrapper.

        Read the `sklearn.impute.SimpleImputer` documentation to get more
        information about the parameters.

        Returns:
            SklearnColumnsWrapper transformer with SimpleImputer as sklearn
            transformer.
        """
        if fill_value is not None:
            strategy = "constant"
        mi = sklearn.impute.SimpleImputer(missing_values=missing_values,
                                          strategy=strategy,
                                          fill_value=fill_value,
                                          verbose=verbose,
                                          copy=copy)
        return SklearnColumnsWrapper(mi, *args, **kwargs)


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
            df = self.transform_group(name, group)
            dfs.append(df)
        df_out = pd.concat(dfs, axis=0)
        return df_out.loc[df_in.index, :]

    def fit_group(self, name, df):
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

