"""'Select' transformer tests. """


from unittest import TestCase

from .context import recipipe as r
from .fixtures import create_df_all


class SelectTest(TestCase):
    """'Select' transformer tests. """

    def _create_fit_transform(self, *args, **kwargs):
        """Create an apply a 'select' on the sample dataframe.

        Args:
            *args: Passed to the 'select'.
            **kwargs: Passed to the 'select'.

        Returns:
            The sample dataframe transformed using a recipipe
            that only contains a 'select'.
        """

        df = create_df_all()
        t = r.recipipe() + r.select(*args, **kwargs)
        return t.fit_transform(df)

    def test_select_one_column(self):
        """Check select works with one column. """

        df = self._create_fit_transform("color")
        expected = ["color"]
        columns = list(df.columns)
        self.assertEqual(columns, expected)

    def test_select_two_columns(self):
        """Check that select works with two columns.

        The order of the returned columns should be the same.
        """

        df = self._create_fit_transform("color", "amount")
        expected = ["color", "amount"]
        columns = list(df.columns)
        self.assertEqual(columns, expected)
