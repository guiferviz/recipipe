
import collections
import copy
import inspect
import re

import numpy as np

import pandas as pd

from sklearn.base import clone

from recipipe.utils import default_params
from recipipe.utils import flatten_list
from recipipe.utils import fit_columns
from recipipe.utils import memory_usage_mb
from recipipe.utils import is_categorical
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

        return df

    def _inverse_transform(self, df_in):
        np_out = self._inverse_transform_columns(df_in, self.cols_out)
        df_out = pd.DataFrame(np_out, columns=self.cols, index=df_in.index)
        return df_out

    def _inverse_transform_columns(self, df, columns_name):  # pragma: no cover
        return df


class CategoryEncoder(ColumnTransformer):

    def __init__(self, *args, error_unknown=False, unknown_value=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.categories = {}
        self.error_unknown = error_unknown
        self.unknown_value = unknown_value

    def _fit_column(self, df, col):
        if is_categorical(df, col):
            cat = df[col].cat.categories
        else:
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

    def _inverse_transform_column(self, df, cols):
        col = cols[0]
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


class SklearnColumnWrapper(ColumnTransformer):

    def __init__(self, sk_transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sk_transformer = sk_transformer
        self.transformers = {}

    def _fit_column(self, df, column_name):
        t = clone(self.sk_transformer)
        t.fit(df[column_name].values.reshape(-1, 1))
        self.transformers[column_name] = t

    def _transform_column(self, df, c):
        return self.transformers[c].transform(df[c].values.reshape(-1, 1))

    def get_column_mapping(self):
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

        col_map = super().get_column_mapping()
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
                if len(new_cols) > 0:
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

    def get_column_mapping(self):
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
            col_map = super().get_column_mapping()
        """
        else:
            name = "_".join(c)
            self.cols_out = [f"{name}_{i}" for i in range(output_size)]
            col_map = {tuple(c): tuple(self.new_cols_list)}
        """
        return col_map


class SklearnFitOneWrapper(SklearnColumnWrapper):
    """Fit in a concatenation of all the columns, apply one by one.

    This is useful when we have two or more columns that are very related.
    For example, if all those columns share the same category type.
    """

    def _fit(self, df):
        one_col = df[self.cols].stack()
        self.sk_transformer.fit(one_col.values.reshape(-1, 1))
        self.transformers = {c: self.sk_transformer for c in self.cols}


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

    WRAPPERS = {
            "column": SklearnColumnWrapper,
            "columns": SklearnColumnsWrapper,
            "fit_one_col": SklearnFitOneWrapper,
    }

    def __init__(self, sk_transformer, **kwargs):
        self.sk_transformer = sk_transformer
        self.r_params = kwargs

    def __call__(self, *args, wrapper="columns", **kwargs):
        """Instantiate a SKLearn wrapper using a copy of the given transformer.

        It's important to make this copy to avoid fitting several times the
        same transformer.
        """

        wrapping_methods = SklearnCreator.WRAPPERS.keys()
        if wrapper not in wrapping_methods:
            raise ValueError("Wrapper method not in {wrapping_methods}")

        # Recipipe transformer params.
        signature = inspect.signature(RecipipeTransformer.__init__)
        params = [i.name for i in signature.parameters.values()]
        r_params = self.r_params.copy()
        for i in params:
            if i in kwargs:
                r_params[i] = kwargs[i]
                del kwargs[i]

        # SKLearn transformer params.
        if "sk_params" in kwargs:
            kwargs = default_params(kwargs, kwargs["sk_params"])
            del kwargs["sk_params"]
        signature = inspect.signature(self.sk_transformer.__init__)
        params = [i.name for i in signature.parameters.values()]
        sk_params = {}
        for i in params:
            if i in kwargs:
                sk_params[i] = kwargs[i]
                del kwargs[i]

        # Create a copy of the sk transformer with the given params.
        t = clone(self.sk_transformer)
        t.set_params(**sk_params)

        return SklearnCreator.WRAPPERS[wrapper](t, *args, **r_params)


class ReplaceTransformer(RecipipeTransformer):

    def __init__(self, *args, values=None, **kwargs):
        assert values is not None
        super().__init__(*args, **kwargs)
        self.values = values
        self.inverse_values = None

    def _fit(self, df):
        self.inverse_values = {v: k for k, v in self.values.items()}

    def _transform(self, df):
        return df[self.cols].replace(self.values)

    def _inverse_transform(self, df):
        return df[self.cols_out].replace(self.inverse_values)


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


class DropNARowsTransformer(RecipipeTransformer):

    def __init__(self, *args, inplace=False, how="any", thresh=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropna_params = dict(axis="index", inplace=inplace)

    def transform(self, df, y=None):
        return df.dropna(subset=self.cols, **self.dropna_params)

    def inverse_transform(self, df):
        return df


class ColumnGroupsTransformer(RecipipeTransformer):
    """Apply a N to 1 transformation to a group of columns. """

    def __init__(self, *args, groups=False, cols_init=None, **kwargs):
        if cols_init:
            args = flatten_list([args, cols_init], recursive=False)
        if groups:
            args = [flatten_list([c]) for c in args]
        self.col_groups_init = args
        self.groups = groups
        super().__init__(*args, **kwargs)

    def _fit(self, df):
        self.col_groups = None
        if self.groups:
            self.col_groups = [fit_columns(df, c) for c in self.col_groups_init]
        if not self.col_groups:
            self.col_groups = [[i] for i in self.cols]

    def get_column_mapping(self):
        col_map = {}
        for c in zip(*self.col_groups):
            new_col = self._get_column_name(c)
            col_map[tuple(c)] = new_col
        return col_map

    def _get_column_name(self, c):
        return self.col_format.format(re.sub(r"\s*\d\s*", "", c[0]))

    def _transform(self, df):
        df_out = pd.DataFrame(index=df.index, columns=self.cols_out)
        for c in zip(*self.col_groups):
            col_out = self.col_map[tuple(c)]
            df_out[col_out] = self._transform_group(df, list(c))
        return df_out

    def _transform_group(self, df, group_cols):  # pragma: no cover
        raise NotImplementedError()

    def _inverse_transform(self, df):
        df_out = pd.DataFrame(index=df.index, columns=self.cols_out)
        for c in zip(*self.col_groups):
            col_out = self.col_map[tuple(c)]
            col_out_value = self._inverse_transform_group(df, col_out)
            for i in c:
                df_out[i] = col_out_value
        return df_out

    def _inverse_transform_group(self, df, col):  # pragma: no cover
        return df[col]


class ReduceMemoryTransformer(RecipipeTransformer):
    """Change data types in order to reduce memory usage.

    This transformer iterates all the columns of the DataFrame and for numeric
    types checks if there is another datatype that occupies less memory and
    can store all the numbers of the column.

    Another way to reduce memory usage of Pandas DataFrame is creating
    categories (:obj:`pandas.Categorical`).
    It's not always possible to save memory by converting to a category.
    The ideal columns to convert to category are those with few different
    values.
    Converting a column that has all unique values (like and ID column) will
    surely occupy more as a category, so consider to drop those columns first
    or not to apply the transformer on those columns.

    In a DataFrame there may also be numeric columns that can be converted to
    categorical (especially columns of integer types).
    You should make those transformations manually if you want to have a 
    column of category dtype from the numeric column.
    Note that the :obj:`recipipe.transformers.CategoryEncoder` is not really
    returning a Pandas category, is encoding the values into integer values.

    You should also note that this transformation is done in place because
    that's the objective of reducing memory.
    Keeping a copy in memory does not make too much sense if you want to save
    memory.
    """

    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    _NUMERIC_TO_REDUCE = ["int16", "int32", "int64", "float64"]

    def __init__(self, *args, deep=True, verbose=False,
            object_to_category=True, **kwargs):
        """Create a new reduce memory transformer.

        Args:
            deep (bool): When `verbose=True` the percentage of memory reduced
                by the conversion is shown on the screen.
                It's computed using the :obj:`pandas.DataFrame.memory_usage`
                method.
                `deep` is the argument of this method.
                `deep=True` is more slow but it's more accurate.
                Note that it's really slow on big DataFrames.
            verbose (bool): Prints informative information about the process on
                the screen.
                Shows the percentage of memory saved after the conversion.
            object_to_category (bool): If `True` this method automatically
                converts objects to the Pandas category type.
        """

        super().__init__(*args, **kwargs)
        self.deep = deep
        self.verbose = verbose
        self.object_to_category = object_to_category
        self.dtypes = None

    def _fit(self, df, y=None):
        dtypes = {}

        # Find a new dtype for each of the columns.
        for col in self.cols:
            col_type = df[col].dtype
            best_type = None
            if col_type == "object":
                if self.object_to_category:
                    best_type = df[col].astype("category").dtype
            elif col_type in ReduceMemoryTransformer._NUMERIC_TO_REDUCE:
                downcast = "integer" if "int" in str(col_type) else "float"
                best_type = pd.to_numeric(df[col], downcast=downcast).dtype

            if best_type is not None and best_type != col_type:
                dtypes[col] = best_type
                # Log the conversion performed.
                if self.verbose:
                    print(f"Column '{col}' of type {col_type} may be better as"
                          f" type {best_type}")

        self.dtypes = dtypes

    def transform(self, df, y=None):
        start_mem = 0
        if self.verbose:
            start_mem = memory_usage_mb(df, deep=self.deep)

        df[:] = df.astype(self.dtypes)

        if self.verbose:
            end_mem = memory_usage_mb(df, deep=self.deep)
            diff_mem = start_mem - end_mem
            percent_mem = 100 * diff_mem / start_mem
            print(f"Memory usage decreased from"
                  f" {start_mem:.2f}MB to {end_mem:.2f}MB"
                  f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")
        return df


class ExtractTransformer(ColumnTransformer):
    """Extract regex from string columns. """

    def __init__(self, *args, pattern=None, flags=0, indicator=False,
                 col_values=None, **kwargs):
        """Create a ExtractTransformer.

        Args:
            pattern (:obj:`list` or :obj:`str`): Regex or list of regex
                to extract. If a list is provided, only one extraction group
                per element is allowed.
            flags (:obj:`int`): Flags from the :obj:`re` module.
                Default: 0, no flags.
            col_values (:obj:`list` of :obj:`str`): Values of the output
                columns. If you are extracting more than one column, from an
                input column `a` and with `col_values=["b", "c"]` you will get
                two output columns named `a=b` and `a=c`.
            indicator (:obj:`bool`): Instead of capturing the string, return
                an indicator with 1 if there is a match and a 0 otherwise.
        """

        super().__init__(*args, **kwargs)

        col_values_none = col_values is None
        if type(pattern) == str:
            r = re.compile(pattern)
            if r.groups == 0:
                # Add a group that captures everything.
                pattern = f"({pattern})"
            if col_values_none:
                col_values = list(map(str, range(max(1, r.groups))))
        elif type(pattern) == list:
            if col_values_none:
                col_values = []
            for i in range(len(pattern)):
                r = re.compile(pattern[i])
                if r.groups > 1:
                    raise ValueError("Only one extraction group per element"
                                     " in the pattern list")
                elif r.groups == 0:
                    # Add a group that captures everything.
                    pattern[i] = f"({pattern[i]})"
                if col_values_none:
                    # Use only alpha-num characters of the expression as name.
                    col_value = re.sub(r"\W+", "", pattern[i])
                    col_values.append(col_value)
                    # TODO: check if duplicate names.
            pattern = "|".join(pattern)

        self.re = re.compile(pattern, flags)
        self.pattern = pattern
        self.flags = flags
        self.indicator = indicator
        self.col_values = col_values

        if self.re.groups != len(col_values):
            raise ValueError("The number of extraction groups should be equal"
                             " to the number of column names")

    def get_column_mapping(self):
        col_format = self.col_format
        if len(self.col_values) > 1:
            if col_format == "{}":
                col_format = "{column}={value}"
        return {
            i: [col_format.format(i, j, column=i, value=j)
                for j in self.col_values]
                    for i in self.cols
        }

    def _transform_column(self, df, c):
        df_out = df[c].str.extract(self.re)
        if self.indicator:
            df_out = df_out.notna().astype("int8")
        return df_out.values


class ConcatTransformer(ColumnGroupsTransformer):
    """Concatenate string or non-string columns into a new string column. """

    def __init__(self, *args, separator="", **kwargs):
        super().__init__(*args, **kwargs)
        self.separator = separator

    def _transform_group(self, df, group_cols):
        return df[group_cols].astype(str).agg(self.separator.join, axis=1)


class SumTransformer(ColumnGroupsTransformer):
    """Sum columns. """

    def _transform_group(self, df, group_cols):
        return df[group_cols].sum(axis=1)


class TargetEncoderTransformer(ColumnTransformer):
    """Target encoder.

    Computed as described in:
    *A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems*, by Daniele Micci-Barreca.

    Code partially taken from:
    https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
    """

    def __init__(self, *args, target=None, min_samples_leaf=1, smoothing=1,
                **kwargs):
        super().__init__(*args, **kwargs)
        if target is None:
            raise ValueError("A target must be specified")
        self.target = target
        self.replace_dicts = {}
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing

    def _fit(self, df):
        if self.target not in df:
            raise ValueError("Target must be in the fitted DataFrame")
        if self.target in self.cols:
            self.cols.remove(self.target)
        super()._fit(df)
        self.inverse_replace_dicts = {k: {v1: k1 for k1, v1 in v.items()}
                for k, v in self.replace_dicts.items()}

    def _fit_column(self, df, col):
        averages = df.groupby(col)[self.target].agg(["mean", "count"])
        smoothing = 1 / (1 + np.exp(-(averages["count"] -
                self.min_samples_leaf) / self.smoothing))
        # Apply average function to all target data
        prior = df[self.target].mean()
        # The bigger the count the less full_avg is taken into account
        output = prior * (1 - smoothing) + averages["mean"] * smoothing
        self.replace_dicts[col] = output.to_dict()

    def _transform_column(self, df, col):
        return df[col].replace(self.replace_dicts[col])

    def _inverse_transform_column(self, df, column_names):
        col_out = column_names[0]
        col_in = self.col_map_1_n[col_out][0]
        return df[col_out].replace(self.inverse_replace_dicts[col_in])


class AsTypeTransformer(RecipipeTransformer):

    def __init__(self, *args, dtypes=None, **kwargs):
        if dtypes is None:
            raise ValueError("dtypes cannot be None")
        self.dtypes = dtypes
        self.original_dtypes = None
        super().__init__(*args, **kwargs)

    def _fit(self, df):
        self.original_dtypes = df[self.cols].dtypes

    def _transform(self, df):
        return df[self.cols].astype(self.dtypes)

    def _inverse_transform(self, df):
        return df[self.cols].astype(self.original_dtypes)

