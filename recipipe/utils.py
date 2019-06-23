
def default_params(fun_kwargs, default_dict=None, **kwargs):
    """Add to kwargs and/or default_dict the values of fun_kwargs. """
    if default_dict is None:
        default_dict = kwargs
    default_dict.update(kwargs)
    default_dict.update(fun_kwargs)
    return default_dict
