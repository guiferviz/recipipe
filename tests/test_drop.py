
from unittest import TestCase

from .context import recipipe as r
from .fixtures import create_df_all


class DropTest(TestCase):

    def _create_fit_transform(self, *args, **kwargs):
        df = create_df_all()
        t = r.recipipe() + r.drop(*args, **kwargs)
        return t.fit_transform(df)

    def test_select_one_column(self):
        """Test if drop can remove one column.

        We do not need more test for drop because it uses the
        same method of fitting column names as all recipipe
        transforms.
        In this test we also check that the order of the remaining
        columns is the same after droping.
        """
        df = self._create_fit_transform("color")
        expected = ["price", "amount"]
        columns = list(df.columns)
        self.assertEqual(columns, expected)
