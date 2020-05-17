"""Improved pipelines for data science projects.

SKLearn pipelines easy to declare and Pandas-compatible.
"""

from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer

from recipipe.core import Recipipe
from recipipe.core import RecipipeTransformer

from recipipe.transformers import CategoryEncoder
from recipipe.transformers import ColumnTransformer
from recipipe.transformers import ColumnsTransformer
from recipipe.transformers import DropTransformer
from recipipe.transformers import DropNARowsTransformer
from recipipe.transformers import GroupByTransformer
from recipipe.transformers import PandasScaler
from recipipe.transformers import QueryTransformer
from recipipe.transformers import ReplaceTransformer
from recipipe.transformers import SelectTransformer
from recipipe.transformers import SklearnCreator
from recipipe.transformers import SklearnColumnsWrapper
from recipipe.transformers import SklearnColumnWrapper

from recipipe._version import __version__


__author__ = "guiferviz"


##############################################
#  Define aliases to make it easier to use.  #
##############################################
recipipe = Recipipe
category = CategoryEncoder
select = SelectTransformer
drop = DropTransformer
dropna = DropNARowsTransformer
dropna_rows = DropNARowsTransformer
query = QueryTransformer
replace = ReplaceTransformer
groupby = GroupByTransformer
# SKLearn recipipe transformer.
from_sklearn = SklearnCreator
onehot = from_sklearn(OneHotEncoder(sparse=False, handle_unknown="ignore"))
scale = from_sklearn(StandardScaler())
binarizer = from_sklearn(Binarizer())
indicator = from_sklearn(MissingIndicator(), col_format="INDICATOR({})")
impute = from_sklearn(SimpleImputer(strategy="constant"))
# Alias for SKLearn transformers.
sk_indicator = MissingIndicator
sk_inputer = SimpleImputer
sk_onehot = OneHotEncoder
sk_scale = StandardScaler
sk_binarizer = Binarizer


def greet():  # pragma: no cover
    """Print a silly sentence to stdout. """

    # https://upjoke.com/recipe-jokes
    print("I've found the recipe for happiness.\n"
          "Can someone just send me some money so I can buy the ingredients?")

