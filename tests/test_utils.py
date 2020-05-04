
from unittest import TestCase

from recipipe.utils import flatten_list


class UtilsTest(TestCase):

    def test_flatten_one_level(self):
        """One level of nested iterators. """

        a = ["c1", ["c2"], set(["c3"]), ("c4", "c5")]
        b = ["c1", "c2", "c3", "c4", "c5"]
        self.assertEqual(flatten_list(a), b)

    def test_flatten_multi_level(self):
        """More than one level of nested iterators, testing recursivity. """

        a = ["c1", ["c2", (["c3", ("c4", [[set(["c5"])], "c6"])])]]
        b = ["c1", "c2", "c3", "c4", "c5", "c6"]
        self.assertEqual(flatten_list(a), b)

