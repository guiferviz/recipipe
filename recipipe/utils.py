
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


def flatten_list(cols_list, recursive=True):
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
            cols.extend(flatten_list(i) if recursive else i)
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

    if not cols and not dtype:
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
        if type(dtype) != dict:
            dtype = dict(include=dtype)
        cols_fitted = list(df.select_dtypes(**dtype).columns)

    if drop_duplicates:
        cols_fitted = list(collections.OrderedDict.fromkeys(cols_fitted))

    return cols_fitted


def add_to_map_dict(col_map, k, v):
    """Stores `k` and `v` in the given `col_map`.

    If `k` is a string, `col_map[k] += list(v)`.
    If `k` is a tuple, `for each i in k: col_map[k] += list(v)`.

    Args:
        col_map (:obj:`dict`): Dictionary in which the keys and values will
            be stored.
        k (:obj:`str` or :obj:`tuple`): Tuples will be split and `col_map`
            will contain a string key with a list of values taken from `v`.
        v (:obj:`str` or :obj:`tuple`): Value appended to each of the keys
            in `k`.
    """

    # Convert v to list, and k to tuple.
    v = [v] if type(v) == str else list(v)
    k = (k,) if type(k) == str else k

    # Iterate the tuple keys and append values.
    for kk in k:
        l = col_map.get(kk, [])
        l += v
        col_map[kk] = l

