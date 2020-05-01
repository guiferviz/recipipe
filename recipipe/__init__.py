"""Improved pipelines for data science projects.

Sklearn pipelines easy to declare and Pandas-compatible.
"""


from recipipe._version import __version__


__author__ = "guiferviz"


def greet():
    """Print a silly sentence to stdout. """

    # https://upjoke.com/recipe-jokes
    print("I've found the recipe for happiness.\n"
          "Can someone just send me some money so I can buy the ingredients?")


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from recipipe.core import CategoryEncoder
from recipipe.core import DropTransformer
from recipipe.core import GroupByTransformer
from recipipe.core import MissingIndicatorCreator
from recipipe.core import PandasScaler
from recipipe.core import QueryTransformer
from recipipe.core import Recipipe
from recipipe.core import ReplaceTransformer
from recipipe.core import SelectTransformer
from recipipe.core import SimpleImputerCreator
from recipipe.core import SklearnCreator


# Aliases to make it easy to use.
recipipe = Recipipe
category = CategoryEncoder
select = SelectTransformer
drop = DropTransformer
from_sklearn = SklearnCreator
onehot = from_sklearn(
    OneHotEncoder(sparse=False, handle_unknown="ignore"),
    keep_original=False)
scale = from_sklearn(
    StandardScaler(),
    keep_original=False)
indicator = MissingIndicatorCreator()
query = QueryTransformer
impute = SimpleImputerCreator()
replace = ReplaceTransformer
groupby = GroupByTransformer

