"""Recipipe.

Recipipe sort description.
"""

from ._version import __version__
from .core import Recipipe
from .core import SScaler
from .core import SklearnCreator
from .core import OneHotEncoder


__author__ = "guiferviz"


def greet():
    """Print a silly sentence. """
    print("An algorithm implemented is worth two in pseudocode.")


recipipe = Recipipe
scale = SScaler
sklearn = SklearnCreator
onehot = sklearn(OneHotEncoder(sparse=False, handle_unknown="ignore"))
