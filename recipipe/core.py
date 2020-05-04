
import abc
import fnmatch

import numpy as np

import pandas as pd

import sklearn
import sklearn.impute
import sklearn.pipeline
from sklearn.base import TransformerMixin, clone

from recipipe.utils import default_params


class Recipipe(sklearn.pipeline.Pipeline):
    """Recipipe pipeline.

    A Recipipe pipeline is an extension of an SKLearn pipeline.
    It adds some functionality that make the creation of pipelines less
    painful.
    For example, the `steps` param is not required at the construction time.
    You can add your transformers to the pipeline anytime using
    :obj:`recipipe.core.Recipipe.add`.

    Attr:
        Same attributes as :obj:`sklearn.pipeline.Pipeline`.
    """

    def __init__(self, steps=None, **kwargs):
        """Create a Recipipe pipeline.

        Args:
            steps (:obj:`list`): Same as in :obj:`sklearn.pipeline.Pipeline`.
            kwargs: Same as in :obj:`sklearn.pipeline.Pipeline`: `memory`
                and `verbose`.
        """

        self.steps = []

        if steps:
            for i in steps:
                self.add(i)
        else:
            # Mock validate steps to avoid empty list validation.
            # This method depends a lot of the hidden representation of the
            # Pipeline class but it's better than using a dummy empty
            # transformer.
            # An empty transformer adds more complexity to make methods like
            # __len__ work properly. Also it's impossible to control the use
            # of self.steps from outside of the class.
            aux = self._validate_steps
            self._validate_steps = lambda: True

        super().__init__(self.steps, **kwargs)

        # Unmock validation method if needed.
        if not steps:
            self._validate_steps = aux

    def __add__(self, transformer):
        """Add a new step to the pipeline using the '+' operator.

        Note that this is exactly the same as calling
        :obj:`recipipe.core.Recipipe.add`.
        The Recipipe object is going to be modified, that is `p = p + t` is
        the same as `p + t`, where `p` is any Recipipe pipeline and `t` is any
        transformer.

        See Also:
            :obj:`recipipe.core.Recipipe.add`
        """

        return self.add(transformer)

    def add(self, step):
        """Add a new step to the pipeline.

        You can add steps even if the pipeline is already fitted, so be
        careful.

        Args:
            step (Transformer or tuple(`str`, Transformer)): The new step that
                you want to add to the pipeline.
                Any transformer is good (SKLearn transformer or
                :obj:`recipipe.core.RecipipeTransformer`).
                If a tuple is given, the fist element of the tuple is going to
                be used as a name for the step in the pipeline.

        Returns:
            The pipeline.
            You can chain `add` methods: `pipe.add(...).add(...)...`.

        See Also:
            :obj:`recipipe.core.Recipipe.__add__`
        """

        if type(step) is not tuple:
            transformer = step
            if getattr(transformer, "name", None):
                name = transformer.name
            else:
                idx = len(self.steps)
                name = f"step{idx:02d}"
            step = (name, transformer)

        self.steps.append(step)

        return self


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
        keep_original (bool): `True` if you want to keep the input columns used
            in the transformer in the transformed DataFrame, `False` if not.
        col_format (str): New name of the columns. Use "{}" in to
            substitute that placeholder by the column name. For example, if
            you want to append the string "_new" at the end of all the
            generated columns you must set `col_format="{}_new"`.

    Attributes:
        name (str): Human-friendly name of the transformer.
    """

    def __init__(self, *args, cols=None, dtype=None, name=None,
                 keep_original=True, col_format="{}"):
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
        self._col_format = col_format
        self.keep_original = keep_original
        self.name = name

    def _fit_cols(self, df: pd.DataFrame):
        """Set :attr:`RecipipeTransformer._cols_fitted` with a value.

        :attr:`RecipipeTransformer._cols_fitted` is the attribute returned by
        :func:`RecipipeTransformer.get_cols`.

        Arguments:
            df (pandas.DataFrame): DataFrame that is been fitted.
        """

        cols_fitted = []
        # If a list of columns is provided to the constructor...
        if self._cols is not None:
            # Check which columns in dataframe match the given columns.
            for i in self._cols:
                cols_match = fnmatch.filter(list(df.columns), i)
                if len(cols_match) == 0:
                    raise ValueError(f"No column match '{i}' in dataframe")
                cols_fitted += cols_match
        # If a data type is specify...
        if self._dtype is not None:
            # Get columns of the given dtype.
            cols_fitted += list(df.select_dtypes(self._dtype).columns)
        # If not cols or dtype given...
        # We check self._cols is None because we want to
        #  allow empty lists in transformers.
        if self._cols is None and len(cols_fitted) == 0:
            # Use all the columns of the fitted dataframe.
            cols_fitted += list(df.columns)
        self._cols_fitted = cols_fitted

    def _get_key_eq_value(self, dict):
        return [k for k, v in dict.items() if k == v]

    def _fit(self, df):
        pass

    def _transform(self, df):
        raise NotImplementedError("Overwrite transform or _transform method")

    def get_cols(self):
        """Returns the list of columns on which the transformer is applied.

        Raises:
            Exception: If the transformer is not fitted yet.
        """

        if self._cols_fitted is None:
            raise Exception("Columns not fitted, call fit before. "
                            "If you get this error from your own Recipipe"
                            "Transformer make sure you are overwriting the"
                            "_fit method an not fit.")
        return self._cols_fitted

    def fit(self, df, y=None):
        """Fit the transformer.

        Args:
            df (pandas.DataFrame): Dataframe used to fit the transformation.
        """

        self._fit_cols(df)
        self._fit(df)
        return self

    def transform(self, df_in):
        in_cols = df_in.columns
        df_out = self._transform(df_in)
        col_map = self.get_column_mapping()
        # If key and value is the same we want to keep the transformed column
        # and remove the input column.
        to_drop = self._get_key_eq_value(col_map)
        df_in = df_in.drop(to_drop, axis=1)
        # Join input columns to output
        df_joined = df_in.join(df_out)
        # Reorder columns.
        ordered_columns = []
        for i in in_cols:
            if i in col_map:
                # FIXME: It's not possible to keep original if we do not rename
                # the output column.
                if self.keep_original and i not in to_drop:
                    ordered_columns.append(i)
                c = col_map[i]
                ordered_columns += c if type(c) == tuple else [c]
            else:
                ordered_columns.append(i)
        return df_joined[ordered_columns]

    def inverse_transform(self, df_in):
        in_cols = df_in.columns
        df_out = self._inverse_transform(df_in)
        col_map = self.get_column_mapping()
        # If key and value is the same we want to keep the transformed column
        # and remove the input column.
        to_drop = self._get_key_eq_value(col_map)
        df_in = df_in.drop(to_drop, axis=1)
        # Join input columns to output
        df_joined = df_in.join(df_out)
        return df_joined

    def get_column_mapping(self):
        """Get the column mapping between the input and transformed dataframe.

        By default it returns a 1:1 map between the input and output columns.
        Make sure your transformed is fitted before calling this function.

        Return:
            A dict in which the key are the input dataframe column names and
            the value is the output dataframe column names.
            Both key and values can be a tuples, tuple:1 useful to indicate that
            one output column has been created from a list of columns from the
            input dataframe, 1:tuple useful to indicate that a list of output
            columns come from one specific column of the input dataframe.
            We use tuples and not list because list are not hashable so they
            cannot be keys in a dict.
        """

        cols = self.get_cols()
        return {i: self._col_format.format(i) for i in cols}

