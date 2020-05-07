"""Improved pipelines for data science projects.

SKLearn pipelines easy to declare and Pandas-compatible.
"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer

from recipipe.core import Recipipe
from recipipe.core import RecipipeTransformer

from recipipe.transformers import CategoryEncoder
from recipipe.transformers import ColumnTransformer
from recipipe.transformers import ColumnsTransformer
from recipipe.transformers import DropTransformer
from recipipe.transformers import GroupByTransformer
from recipipe.transformers import MissingIndicatorCreator
from recipipe.transformers import PandasScaler
from recipipe.transformers import QueryTransformer
from recipipe.transformers import ReplaceTransformer
from recipipe.transformers import SelectTransformer
from recipipe.transformers import SimpleImputerCreator
from recipipe.transformers import SklearnCreator

from recipipe._version import __version__


__author__ = "guiferviz"

##############################################
#  Define aliases to make it easier to use.  #
##############################################
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
binarizer = from_sklearn(
    Binarizer(),
    keep_original=False)
indicator = MissingIndicatorCreator()
query = QueryTransformer
impute = SimpleImputerCreator()
replace = ReplaceTransformer
groupby = GroupByTransformer


def greet():
    """Print a silly sentence to stdout. """

    # https://upjoke.com/recipe-jokes
    print("I've found the recipe for happiness.\n"
          "Can someone just send me some money so I can buy the ingredients?")

