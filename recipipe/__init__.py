"""Recipipe.

Improved pipelines for data science projects.
"""

from sklearn.preprocessing import OneHotEncoder

from ._version import __version__
from .core import CategoryEncoder
from .core import DropTransformer
from .core import Recipipe
from .core import SelectTransformer
from .core import SklearnScaler
from .core import SklearnCreator


__author__ = "guiferviz"


def greet():
    """Print a famous quote of my friend Albert. """
    print('"If I had my life to live over again, I\'d be a plumber."'
          ' - Albert Einstein')


# Aliases to make it easy to use.
recipipe = Recipipe
category = CategoryEncoder
select = SelectTransformer
drop = DropTransformer
scale = SklearnScaler
from_sklearn = SklearnCreator
onehot = from_sklearn(OneHotEncoder(sparse=False, handle_unknown="ignore"), keep_original=False)
