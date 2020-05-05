
from unittest import TestCase

from tests.fixtures import TransformerMock
from tests.fixtures import RecipipeTransformerMock

import recipipe as r


class RecipipeTest(TestCase):

    def test_no_error_empty_init(self):
        """Test the pipeline constructor.

        No expecting any error with empty pipelines.
        """

        r.recipipe()

    def test_error_empty_fit(self):
        """An exception should be throw when trying to fit empty pipelines.

        The type of the exception is not important.
        """

        p = r.recipipe()
        with self.assertRaises(Exception):
            p.fit(None)  # We check later that fit None should work.

    def test_with_transformers(self):
        """Constructor with transformers. """

        t = TransformerMock()
        p = r.recipipe([t, t])

    def test_len_empty(self):
        """Len of pipeline should give 0 when no transformers provided. """

        p = r.recipipe()
        self.assertEqual(len(p), 0)

    def test_len_with_transformers(self):
        """Len of pipeline should give the number of transformers provided. """

        t = TransformerMock()
        p = r.recipipe([t, t])
        self.assertEqual(len(p), 2)

    def test_transformer_name_generic(self):
        """The steps should have a default name indicating the step index. """

        t0 = TransformerMock()
        t1 = TransformerMock()
        p = r.recipipe([t0, t1])
        self.assertEqual(p.steps[0][0], "step00")
        self.assertEqual(p.steps[1][0], "step01")
        self.assertEqual(p.steps[0][1], t0)
        self.assertEqual(p.steps[1][1], t1)

    def test_transformer_name_attribute(self):
        """The steps should have transformer attr "name" as key. """

        t0 = TransformerMock()
        t0.name = "t0_name"
        t1 = TransformerMock()
        t1.name = "t1_name"
        p = r.recipipe([t0, t1])
        print(p.steps)
        self.assertEqual(p.steps[0][0], "t0_name")
        self.assertEqual(p.steps[1][0], "t1_name")
        self.assertEqual(p.steps[0][1], t0)
        self.assertEqual(p.steps[1][1], t1)

    def test_add_after_creation(self):
        """Check the add method. """

        t = TransformerMock()
        p = r.recipipe()
        p.add(t)
        self.assertEqual(p.steps[0][0], "step00")
        self.assertEqual(p.steps[0][1], t)

    def test_add_after_creation_operator(self):
        """Check the + operator. """

        t = TransformerMock()
        p = r.recipipe()
        p + t
        self.assertEqual(p.steps[0][0], "step00")
        self.assertEqual(p.steps[0][1], t)

    def test_fit_and_transform(self):
        """Test that all the transformer fit/transform methods are called. """

        t = TransformerMock()
        p = r.recipipe([t, t, t])
        p.fit(None)
        p.transform(None)
        # Let's call n = number of steps of the pipeline.
        n = 3
        # The fit method is called n times.
        self.assertEqual(t.n_fit, n)
        # Called n * 2 - 1. Pipelines use the output of the previous step
        # as input to the next one, so the transform method should be called
        # while fitting to know which input the next step is going to receive.
        # The last transformer does not need to be used while fitting because
        # his output is not required by any other transformer, so - 1.
        # Of course, the p.transform call adds n to the number of
        # transformations performed.
        self.assertEqual(t.n_transform, n * 2 - 1)


class RecipipeTransformerTest(TestCase):

    def test_abstract_class(self):
        """You cannot instantiate a RecipipeTransformer. """

        with self.assertRaises(TypeError):
            r.RecipipeTransformer()

    def test_inheritance(self):
        """You should implement the _transform method in any subclass. """

        class TestTransformer(r.RecipipeTransformer):
            def _transform(self, df):
                pass

        TestTransformer()

    def test_init_cols_mix(self):
        t = RecipipeTransformerMock(cols=[
            "c1", ["c2"], set(["c3"]), ("c4", "c5")])
        self.assertEqual(len(t.cols_init), 5)

    def test_init_args_mix(self):
        """Strs, lists, sets and tuples are allowed as var args. """

        t = RecipipeTransformerMock("c1", ["c2"], set(["c3"]), ("c4", "c5"))
        self.assertEqual(len(t.cols_init), 5)

    def test_init_cols_args(self):
        """Cols is appended to args. """

        t = RecipipeTransformerMock("c1", cols=["c2"])
        self.assertListEqual(t.cols_init, ["c1", "c2"])

