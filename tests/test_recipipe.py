
from unittest import TestCase

from .context import recipipe as r
from .fixtures import create_df_all


class RecipipeTest(TestCase):

    def test_error_empty_recipipe(self):
        """An error should be throw when you try to fit an empty recipipe. """
        df = create_df_all()
        t = r.recipipe()
        with self.assertRaises(Exception):
            t.fit(df)
