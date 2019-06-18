
import abc

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer


class Recipipe(TransformerMixin):
    """Recipipe pipeline. """

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

    def get_step(self, name):
        return self.get_step_dict()[name]

    def get_pipeline(self):
        return self.pipeline

    def fit(self, df):
        self.pipeline = self._create_pipeline()
        self.pipeline.fit(df)
        return self

    def transform(self, df):
        return self.pipeline.transform(df)


class RecipipeTransformer(TransformerMixin):
    """Base class of all Recipipe transformers.

    `*args`, `cols` and `dtype` are mutually exclusive.

    Args:
        *args (List[str]): List of column names to apply transformer to.
        cols (List[str]): List of column names to apply transformer to.
        dtype (dtype, str, List[dtype] or List[str]): This value is passed to
            `pandas.DataFrame.select_dtypes`. The columns returned by this
            pandas method (executed in the dataframe passed to the fit method)
            will be the columns that are going to be used in the transformation.
        name (str): Human-friendly name of the transformer.

    Attributes:
        name (str): Human-friendly name of the transformer.
    """

    def __init__(self, *args, cols=None, dtype=None, name=None):
        # Variable args or cols, but not both.
        assert not(cols is not None and args)
        # Set cols using variable args.
        if len(args) > 0:
            if type(args[0]) == list:
                cols = args[0]
            else:
                cols = args
        # Or cols or dtype, but not both.
        assert not(cols is not None and dtype is not None)
        # Set values.
        self._cols = list(cols) if cols is not None else None
        self._dtype = dtype
        self._cols_fitted = None
        self.name = name

    def _fit_cols(self, df: pd.DataFrame):
        """Set :attr:`RecipipeTransformer._cols_fitted` with a value.

        :attr:`RecipipeTransformer._cols_fitted` is the attribute returned by
        :func:`RecipipeTransformer.get_cols`.

        Arguments:
            df (pandas.DataFrame): DataFrame that is been fitted.
        """

        # If a list of columns is provided to the constructor...
        if self._cols is not None:
            self._cols_fitted = self._cols
        # If a data type is specify...
        elif self._dtype is not None:
            # Get columns of the given dtype.
            self._cols_fitted = list(df.select_dtypes(self._dtype).columns)
        # If not cols or dtype given...
        else:
            # Use all the columns of the fitted dataframe.
            self._cols_fitted = list(df.columns)

    def get_cols(self):
        """Returns the list of columns on which the transformer is applied.

        Raises:
            Exception: If the transformer is not fitted yet.
        """

        if self._cols_fitted is None:
            raise Exception("Columns not fitted, call fit before. "
                            "Make sure you call super().fit(df) on "
                            "your transformer subclass.")
        return self._cols_fitted

    def fit(self, df):
        """Fit the transformer.

        Args:
            df (pandas.DataFrame): Dataframe used to fit the transformation.
        """

        self._fit_cols(df)


class SelectTransformer(RecipipeTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, df, y=None):
        super().fit(df)
        return self

    def transform(self, df):
        cols = self.get_cols()
        return df[cols]


class DropTransformer(RecipipeTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, df, y=None):
        super().fit(df)
        return self

    def transform(self, df):
        cols = self.get_cols()
        return df.drop(cols, axis=1)


class ColumnTransformer(RecipipeTransformer):

    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, df, y=None):
        super().fit(df)
        for i in self.get_cols():
            self._fit_column(df, i)
        return self

    @abc.abstractmethod
    def _fit_column(self, df, column_name):
        pass

    def transform(self, df):
        for i in self.get_cols():
            df[i] = self._transform_column(df, i)
        return df

    @abc.abstractmethod
    def _transform_column(self, df, column_name):
        pass


class ColumnsTransformer(RecipipeTransformer):

    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, df, y=None):
        super().fit(df)
        c = self.get_cols()
        self._fit_columns(df, c)
        return self

    @abc.abstractmethod
    def _fit_columns(self, df, column_name):
        pass

    def transform(self, df):
        c = self.get_cols()
        df[c] = self._transform_columns(df, c)
        return df

    @abc.abstractmethod
    def _transform_columns(self, df, column_name):
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


class SklearnScaler(ColumnsTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ss = StandardScaler()

    def _fit_columns(self, df, c):
        self.ss.fit(df[c].values)

    def _transform_columns(self, df, c):
        return self.ss.transform(df[c].values)


class PandasScaler(ColumnTransformer):
    """Standard scaler implemented with Pandas operations. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = {}
        self.stds = {}

    def _fit_column(self, df, c):
        self.means[c] = df[c].mean()
        self.stds[c] = df[c].std()

    def _transform_column(self, df, c):
        return (df[c] - self.means[c]) / self.stds[c]


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

    def __init__(self, sklearn_transformer):
        self.trans = sklearn_transformer

    def __call__(self, *args, **kwargs):
        """Instantiate a SklearnWrapper using a copy of the given transformer.

        It's important to make this copy to avoid fitting several times the
        same transformer.
        """
        return SklearnWrapper(clone(self.trans), *args, **kwargs)


class SklearnWrapper(RecipipeTransformer):

    def __init__(self, sklearn_transformer, *args, cols=None,
                 keep_cols=False, separator="=", quote=True):
        super().__init__(*args, cols=cols)
        self.sklearn_transformer = sklearn_transformer
        self.new_cols = None
        self.new_cols_hierarchy = None
        self.separator = separator
        self.quote = quote
        self.keep_cols = keep_cols

    def fit(self, df, y=None):
        # TODO: Refactor!!!!!!!
        super().fit(df)
        c = self.get_cols()
        self.sklearn_transformer.fit(df[c].values)
        if hasattr(self.sklearn_transformer, "get_feature_names"):
            # c_aux to avoid splitting column names that contain "_".
            c_aux = [str(i) for i in range(len(c))]
            str_format = "{}{}'{}'" if self.quote else "{}{}{}"
            new_cols = self.sklearn_transformer.get_feature_names(c_aux)
            new_cols = [i.split("_", 1) for i in new_cols]
            new_cols = [(c[int(i[0])], i[1].replace("'", "\\'")) for i in new_cols]
            new_cols_hierarchy = {i[0]: [] for i in new_cols}
            for k, v in new_cols:
                new_cols_hierarchy[k].append(v)
            new_cols = [str_format.format(i[0], self.separator, i[1]) for i in new_cols]
            self.new_cols = new_cols
            return_cols = []
            for i in df.columns:
                if i in new_cols_hierarchy:
                    if self.keep_cols:
                        return_cols.append(i)
                    return_cols += [str_format.format(i, self.separator, j) for j in new_cols_hierarchy[i]]
                else:
                    return_cols.append(i)
            self.return_cols = return_cols
        else:
            self.new_cols = ["{}_{}".format("_".join(c), i) for i in range(output.shape[1])]
        return self

    def transform(self, df):
        c = self.get_cols()
        output = self.sklearn_transformer.transform(df[c].values)
        df_new = pd.DataFrame(output, columns=self.new_cols)
        df_new_joined = df.join(df_new)
        return df_new_joined[self.return_cols]

    def _columns_in_order(self, cols, new_cols):
        pass

from sklearn.base import clone
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


def fun(f, **kwargs):
    return SklearnCreator(FunctionTransformer(func=f, **kwargs))
