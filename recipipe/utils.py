
import fnmatch
import collections


def default_params(fun_kwargs, default_dict=None, **kwargs):
    """Add to kwargs and/or default_dict the values of fun_kwargs.
    
    This function allows the user to overwrite default values of some
    parameters. For example, in the next example the user cannot give a value
    to the param `a` because you will be passing the param `a` twice to the
    function `another_fun`::

        >>> def fun(**kwargs):
        ...     return another_fun(a="a", **kwargs)

    You can solve this in two ways. The fist one::

        >>> def fun(a="a", **kwargs):
        ...     return another_fun(a=a, **kwargs)
 
    Or using default_params::

        >>> def fun(**kwargs):
        ...    kwargs = default_params(kwargs, a="a")
        ...    return another_fun(**kwargs)
    """

    if default_dict is None:
        default_dict = kwargs
    else:
        default_dict.update(kwargs)
    default_dict.update(fun_kwargs)
    return default_dict


def reduce_memory_usage(df, deep=True, verbose=True,
                        object_to_category=True):
    """Change datatypes in order to reduce memory usage.

    This function iterates all the columns of the dataframe and
    for numeric types checks if there is another datatype that
    occupies less memory and can store all the numbers of the
    column.

    Another way to reduce memory usage of Pandas dataframes is
    creating categories. This function in verbose mode shows
    if there are columns of type object that could be converted
    to category. It is not always possible to save memory by
    converting to a category. The ideal columns to convert to
    category are those with few different values. Converting a
    column that has all unique values (like and ID column) will
    surely occupy more as a category.
    In a dataframe there may also be numeric columns that can be
    converted to categorical (especially columns of integer
    types). This category conversions must be done by the user.

    All the conversions are made in place, so the original dataframe
    will be overwritten.

    Args:
        df (pandas.DataFrame): Dataframe to change data types.
        deep (bool): When `verbose=True` the percentage of memory reduced
            by the conversion is shown of the screen. It's computed
            using the `pandas.DataFrame.memory_usage` method. `deep` is
            an argument of this method. `deep=True` is more slow but it's
            more accurate.
        verbose (bool): Prints informative information about the process on
            the screen.
        object_to_category (bool): If `True` this method automatically converts
            object to the pandas category type.
    """
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            if object_to_category:
                df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' of type {col_type} may be better as "
                  f"type {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")
    return df


def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """

    return df.memory_usage(*args, **kwargs).sum() / 1024**2


def is_categorical(s, column=None):
    """Check if a pandas Series or a column in a DataFrame is categorical.
    
    s (pandas.Series or pandas.DataFrame): Series to check or dataframe with
        column to check.
    column (str): Column name. If a column name is given it is assumed that
        `s` is a `pandas.DataFrame`.
    """

    if column is not None:
        # s is not a Series, it is a DataFrame.
        s = s[column]
    return s.dtype.name == "category"


def flatten_list(cols_list):
    """Take a list of lists and return a flattened list.

    Args:
        cols_list: an iterable of any quantity of str/tuple/list/set.

    Example:

        >>> flatten_list(["a", ("b", set(["c"])), [["d"]]])
        ["a", "b", "c", "d"]
    """

    cols = []
    for i in cols_list:
        if isinstance(i, (set, list, tuple)):
            cols.extend(flatten_list(i))  # Flatten all the levels recursively.
        else:
            cols.append(i)
    return cols


def fit_columns(df, cols=None, dtype=None, raise_error=True,
        drop_duplicates=True):
    """Fit columns to a DataFrame.

    If no `cols` and not `dtype` are given, `df.columns` is returned.
    If both `cols` and `dtype` are given, first `cols` are applied and `dtype`
    is applied over the resulting columns.

    Note than an empty list can be returned if `df` does not contain columns.

    Args:
        df (:obj:`pandas.DataFrame`): DataFrame that is been fitted.
        cols (:obj:`list`): List of columns to fit. The names may contain
            Unix filename pattern matching (:obj:`fnmatch`) symbols.
        dtype: Any value suported by :obj:`pandas.DataFrame.select_dtypes`.
        raise_error (:obj:`bool`): If `True` and not column in `df` match the
            given column in `cols`, an exception is raised.
        drop_duplicates (:obj:`bool`): Remove duplicates keeping the order.
            Default: `True`.

    Returns:
        List of existing columns in df that satisfy the constrains of `dtype`
        and `cols`.

    Raises:
        :obj:`ValueError` if the patter in `cols` does not match any column and
        `raise_error` is set to `True`.
    """

    cols_fitted = []
    df_cols = list(df.columns)

    if cols is None and dtype is None:
        return df_cols

    if cols:
        for i in cols:
            cols_match = fnmatch.filter(df_cols, i)
            if raise_error and not cols_match:
                raise ValueError(f"No column match '{i}' in dataframe")
            cols_fitted += cols_match

    if dtype:
        if cols_fitted:
            # dtype is applied after cols.
            df = df[cols_fitted]
        cols_fitted = list(df.select_dtypes(dtype).columns)

    if drop_duplicates:
        cols_fitted = list(collections.OrderedDict.fromkeys(cols_fitted))

    return cols_fitted


def get_keys_eq_value(dic):
    """Return a list of keys that are equal to their value.

    Args:
        dic (:obj:`dict`): Dictionary.
    """

    return [k for k, v in dic.items() if k == v]

