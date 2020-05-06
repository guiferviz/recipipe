
import abc
import collections
import fnmatch
import inspect

import sklearn.pipeline
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from recipipe.utils import add_to_map_dict
from recipipe.utils import flatten_list
from recipipe.utils import fit_columns


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


class RecipipeTransformer(BaseEstimator, TransformerMixin, abc.ABC):
    """Base class of all Recipipe transformers.

    Attributes:
        name (str): Human-friendly name of the transformer.
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator.

        Taken from SKLearn with some extra modifications to allow the use of
        var positional arguments (*args) in estimators.
        My modifications are indicated by comments with two ##.
        """

        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:  # pragma: no cover
            ## I don't know how to force this situation for testing...
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD
                                          ## No positional vars (*args).
                                          and p.kind != p.VAR_POSITIONAL]
        ## No RuntimeError raised here!
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def __init__(self, *args, cols=None, dtype=None, name=None,
                 keep_original=False, col_format="{}",
                 cols_not_found_error=False, cols_remove_duplicates=True):
        """Create a new transformer.

        Columns names can be use Unix filename pattern matching (
        :obj:`fnmatch`).

        Args:
            *args (:obj:`list` of :obj:`str`): List of columns the transformer
                will work on.
            cols (:obj:`list` of :obj:`str`):  List of columns the transformer
                will work on. If `*args` are provided, the column array is
                going to be appended at the end.
            dtype (dtype, str, list[dtype] or list[str]): This value is passed
                to :obj:`pandas.DataFrame.select_dtypes`. The columns returned
                by this method (executed in the dataframe passed to the fit
                method) will be the columns that are going to be used in the
                transformation phase.
            name (:obj:`str`): Human-friendly name of the transformer.
            keep_original (:obj:`bool`): `True` if you want to keep the input
                columns used in the transformer in the transformed DataFrame,
                `False` if not.
                Note that, if the output column has the same name as the input
                column, the output input column will not be included even if
                `keep_original` is set to `True`.
                Default: `False`.
            col_format (:obj:`str`): New name of the columns. Use "{}" in to
                substitute that placeholder by the column name. For example, if
                you want to append the string "_new" at the end of all the
                generated columns you must set `col_format="{}_new"`.
                Default: "{}".
            cols_not_found_error (:obj:`bool`): Raise an error if the isn't
                any match for any of the specified columns.
                Default: `False`.
            cols_remove_duplicates (:obj:`bool`): Remove duplicates if a column
                is specified more than once in `args` or `cols`.
                Default: `True`.
        """

        if cols:
            args = (args, cols)
        cols = flatten_list(args)

        # Set values.
        self.cols_init = cols
        self.dtype = dtype
        self.keep_original = keep_original
        self.name = name
        self.col_format = col_format
        self.col_map = None  # set in fit
        self.cols = None  # fitted columns, set in fit
        self.cols_out = None  # set in fit
        self.cols_not_found_error = cols_not_found_error
        self.cols_remove_duplicates = cols_remove_duplicates
        self.cols_to_drop = None  # set in fit

    def _fit(self, df):  # pragma: no cover
        """Your fit code should be here.

        Args:
            df (:obj:`pandas.DataFrame`): DataFrame used for fitting.
        """

        pass

    @abc.abstractmethod
    def _transform(self, df):  # pragma: no cover
        """Your transform code should be here.

        Abstract method that you should overwrite in your classes.
        Remember that any transformation done here should be consistent with
        the column mapping returned by
        :obj:`recipipe.core.RecipipeTransformer._get_column_mapping`.
        Ex: if your column mapping is `{"c1": "c1"}` do not return in this
        method a DataFrame with columns `c1` and `c2`.

        Args:
            df (:obj:`pandas.DataFrame`): DataFrame to transform.

        Return:
            The transformer DataFrame.
        """

        pass

    def fit(self, df, y=None):
        """Fit the transformer.

        Args:
            df (pandas.DataFrame): Dataframe used to fit the transformation.
        """

        self.cols = fit_columns(df, self.cols_init, self.dtype,
                self.cols_not_found_error, self.cols_remove_duplicates)
        self._fit(df)
        # Save column maps and lists.
        self.col_map = self._get_column_mapping()
        col_map_1_n, col_map_1_n_inverse = {}, {}
        for k, v in self.col_map.items():
            add_to_map_dict(col_map_1_n, k, v)
            add_to_map_dict(col_map_1_n_inverse, v, k)
        self.col_map_1_n = col_map_1_n
        self.col_map_1_n_inverse = col_map_1_n_inverse
        self.cols_out = flatten_list(self.col_map.values())
        # Cols in input columns and output columns should be removed from
        # df_in during the transform phase.
        # We join df_in with df_out, so we do not want duplicate column names.
        self.cols_to_drop = set(self.cols).intersection(set(self.cols_out))

        return self

    def transform(self, df_in):
        """Transform DataFrame.

        Args:
            df_in (:obj:`pandas.DataFrame`): Input DataFrame.

        Returns:
            Transformed DataFrame.

        Raise:
            :obj:`ValueError` if not `cols` fitted. Fit the transform to avoid
            this error.
        """

        if self.cols is None:
            raise ValueError("No cols set. Transformer not fitted?")

        in_cols = df_in.columns
        df_out = self._transform(df_in)
        df_in = df_in.drop(self.cols_to_drop, axis=1)
        # Join input columns to output
        df_joined = df_in.join(df_out)

        # Reorder output columns and return.
        # This cannot be precomputed during fit because we want to support
        # any extra column not present during fit time. We also want to
        # maintain the input column order.
        # Ex: we can fit a df that contains a target column and transform dfs
        # without that target.
        cols_out = self._get_ordered_out_cols(in_cols)

        return df_joined[cols_out]

    def _get_ordered_out_cols(self, in_cols):
        cols_out = []
        for i in in_cols:
            if i in self.cols:
                # It's not possible to keep original if we do not rename
                # the output column.
                if self.keep_original and i not in self.cols_to_drop:
                    cols_out.append(i)
                cols_out += self.col_map_1_n[i]
            else:
                cols_out.append(i)
        # Remove duplicates.
        cols_out = list(collections.OrderedDict.fromkeys(cols_out))
        return cols_out

    def inverse_transform(self, df_in):
        in_cols = df_in.columns
        df_out = self._inverse_transform(df_in)
        to_drop = []
        cols_out = []
        for i in in_cols:
            l = self.col_map_1_n_inverse.get(i, None)
            if l:
                cols_out += l
            if i in self.cols_out:
                to_drop.append(i)
            else:
                cols_out.append(i)
        df_in = df_in.drop(to_drop, axis=1)
        df_joined = df_in.join(df_out)
        # Remove duplicates.
        cols_out = list(collections.OrderedDict.fromkeys(cols_out))
        return df_joined[cols_out]

    def _get_column_mapping(self):
        """Get the column mapping between the input and transformed DataFrame.

        By default it returns a 1:1 map between the input and output columns.
        Make sure your transformer is fitted before calling this function.

        Return:
            A dict in which the keys are the input DataFrame column names and
            the value is the output DataFrame column names.
            Both key and values can be tuples, tuple:1 useful to indicate that
            one output column has been created from a list of columns from the
            input DataFrame, 1:tuple useful to indicate that a list of output
            columns come from one specific column of the input DataFrame.
            We use tuples and not lists because lists are not hashable, so they
            cannot be keys in a dict.

        See Also:
            :obj:`recipipe.core.RecipipeTransformer.col_format`

        Raise:
            :obj:`ValueError` if `self.cols` is `None`.
        """

        if self.cols is None:
            raise ValueError("No columns. Transformer not fitted?")

        return {i: self.col_format.format(i) for i in self.cols}

