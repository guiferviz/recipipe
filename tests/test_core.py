
from unittest import TestCase
from unittest.mock import MagicMock

from tests.fixtures import TransformerMock
from tests.fixtures import RecipipeTransformerMock
from tests.fixtures import create_df_3dtypes

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

    def test_inheritance(self):
        """Inherit and test constructor. """

        class TestTransformer(r.RecipipeTransformer):
            pass

        TestTransformer()

    def test_inheritance_var_args_sklearn_params_no_init(self):
        class T(r.RecipipeTransformer):
            pass

        print(T(keep_original=False))

    def test_inheritance_var_args_sklearn_params(self):
        """Params are used as SKLearn estimator params (they are inherit). """

        class T1(r.RecipipeTransformer):
            def __init__(self, *args, param1=1, **kwargs):
                self.param1 = param1
                super().__init__(*args, **kwargs)
        class T2(T1):
            def __init__(self, *args, param1=7, param2=2, **kwargs):
                self.param2 = param2
                super().__init__(*args, param1=param1, **kwargs)

        params = T2(1, 2, param1=3, param2=4, name="The Dude").get_params()
        params_expected = dict(param1=3, param2=4, name="The Dude",
                col_format='{}', cols_init=[1, 2], cols_not_found_error=False,
		dtype=None, keep_original=False)
        self.assertDictEqual(params, params_expected)

    def test_init_cols_mix(self):
        t = RecipipeTransformerMock(cols_init=[
            "c1", ["c2"], set(["c3"]), ("c4", "c5")])
        self.assertEqual(len(t.cols_init), 5)

    def test_init_args_mix(self):
        """Strs, lists, sets and tuples are allowed as var args. """

        t = RecipipeTransformerMock("c1", ["c2"], set(["c3"]), ("c4", "c5"))
        self.assertEqual(len(t.cols_init), 5)

    def test_init_cols_args(self):
        """Cols is appended to args. """

        t = RecipipeTransformerMock("c1", cols_init=["c2"])
        self.assertListEqual(t.cols_init, ["c1", "c2"])

    def test_fit_cols(self):
        """Cols should have a value after fit. """

        t = RecipipeTransformerMock("c*", dtype=int)
        t.fit(create_df_3dtypes())
        self.assertListEqual(t.cols, ["c1"])
        self.assertEqual(t.n_fit, 1)

    def test_fit_cols_all(self):
        """When not cols are specified we need to fit all of them. """

        t = RecipipeTransformerMock()
        t.fit(create_df_3dtypes())
        self.assertListEqual(t.cols, ["c1", "c2", "t1"])
        self.assertEqual(t.n_fit, 1)

    def test_fit_cols_keep_original_collision(self):
        """Keep original only works when no name collisions exist. """

        t = RecipipeTransformerMock(keep_original=True)
        with self.assertRaises(ValueError):
            t.fit(create_df_3dtypes())

    # TODO: this test is not working, do we really need to check the returned
    # column mapping?
    def _test_fit_check_column_mapping(self):
        t = r.RecipipeTransformer()
        t.get_column_mapping = MagicMock(return_value={
            "c1": ["c1", "c2"], "c2": []})
        with self.assertRaises(ValueError):
            t.fit(create_df_3dtypes())

    def test_get_column_map_not_fitted(self):
        """Error in column map if no columns are fitted. """

        t = RecipipeTransformerMock()
        with self.assertRaises(ValueError):
            t.get_column_mapping()

    def test_get_column_map(self):
        """Default column mapping, 1:1 mapping. """

        t = RecipipeTransformerMock()
        t.cols = ["c2", "c1"]
        cols_map = t.get_column_mapping()
        self.assertDictEqual(cols_map, {"c2": "c2", "c1": "c1"})

    def test_get_column_map_format(self):
        """Column mapping should use `cols_format`. """

        t = RecipipeTransformerMock(col_format="{}_new")
        t.cols = ["c2", "c1"]
        cols_map = t.get_column_mapping()
        self.assertDictEqual(cols_map, {"c2": "c2_new", "c1": "c1_new"})

    def test_transform_all_columns(self):
        """Transform a df and return the same columns. """

        t = RecipipeTransformerMock()
        df = create_df_3dtypes()
        t.fit(df)
        df = t.transform(df)
        self.assertListEqual(list(df.columns), ["c1", "c2", "t1"])

    def test_transform_some_columns(self):

        class C(r.RecipipeTransformer):
            def _transform(self, df):
                return df[self.cols]
        t = C("c1", "c2")
        df = create_df_3dtypes()
        t.fit(df)
        df = t.transform(df)
        self.assertListEqual(list(df.columns), ["c1", "c2", "t1"])

    def test_transform_keep_original(self):

        class C(r.RecipipeTransformer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def _transform(self, df):
                df = df[self.cols]
                df.columns = [self.col_format.format(i) for i in df.columns]
                return df
        t = C("c1", "c2", keep_original=True, col_format="{}_out")
        df = create_df_3dtypes()
        t.fit(df)
        df = t.transform(df)
        out_cols = ["c1", "c1_out", "c2", "c2_out", "t1"]
        self.assertListEqual(list(df.columns), out_cols)

    def test_transform_keep_original_false_and_format(self):

        class C(r.RecipipeTransformer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def _transform(self, df):
                df = df[self.cols]
                df.columns = [self.col_format.format(i) for i in df.columns]
                return df
        t = C("c1", "c2", keep_original=False, col_format="{}_out")
        df = create_df_3dtypes()
        t.fit(df)
        df = t.transform(df)
        out_cols = ["c1_out", "c2_out", "t1"]
        self.assertListEqual(list(df.columns), out_cols)

    def test_transform_cols_map_tuples(self):

        class C(r.RecipipeTransformer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def get_column_mapping(self):
                return {"c1": ("c1_1", "c1_2"), ("c1", "t1"): "c1t1"}
            def _transform(self, df):
                df = df[["c1", "c1", "t1"]]
                df.columns = ["c1_1", "c1_2", "c1t1"]
                return df
        t = C("c1", "t1")
        df = create_df_3dtypes()
        t.fit(df)
        df = t.transform(df)
        out_cols = ["c1_1", "c1_2", "c1t1", "c2"]
        self.assertListEqual(list(df.columns), out_cols)

    def test_transform_cols_map_str_and_tuples(self):
        """Test 1:1 and n:1 in the same map. """

        class C(r.RecipipeTransformer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def get_column_mapping(self):
                return {"c1": "c1", ("c1", "t1"): "c1t1"}
            def _transform(self, df):
                df = df[["c1", "t1"]]
                df.columns = ["c1", "c1t1"]
                return df
        t = C("c1", "t1")
        df = create_df_3dtypes()
        t.fit(df)
        df = t.transform(df)
        out_cols = ["c1", "c1t1", "c2"]
        self.assertListEqual(list(df.columns), out_cols)

    def test_cols_taken_from_col_map(self):
        """If no cols are given, the col_map should be used to obtain them. """

        class C(r.RecipipeTransformer):
            def get_column_mapping(self):
                return {"c1": ["hi", "bye"]}

        t = C()
        t.fit(create_df_3dtypes())
        self.assertListEqual(t.cols, ["c1"])

    def test_inverse_transform_cols_map_tuples(self):

        class C(r.RecipipeTransformer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def get_column_mapping(self):
                return {"c1": ("c1_1", "c1_2"), ("c1", "t1"): "c1t1"}
            def _transform(self, df):
                df = df[["c1", "c1", "t1"]]
                df.columns = ["c1_1", "c1_2", "c1t1"]
                return df
            def _inverse_transform(self, df):
                df = df[["c1_1", "c1t1"]]
                df.columns = ["c1", "t1"]
                return df
        t = C("c1", "t1")
        df = create_df_3dtypes()
        t.fit(df)
        df = t.transform(df)
        df = t.inverse_transform(df)
        out_cols = ["c1", "t1", "c2"]
        self.assertListEqual(list(df.columns), out_cols)

    def test_inverse_transform_cols_map_str_and_tuples(self):
        """Test 1:1 and n:1 in the same map. """

        class C(r.RecipipeTransformer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def get_column_mapping(self):
                return {"c1": "c1", ("c1", "t1"): "c1t1"}
            def _transform(self, df):
                df = df[["c1", "t1"]]
                df.columns = ["c1", "c1t1"]
                return df
            def _inverse_transform(self, df):
                df = df[["c1", "c1t1"]]
                df.columns = ["c1", "t1"]
                return df
        t = C("c1", "t1")
        df = create_df_3dtypes()
        t.fit(df)
        df = t.transform(df)
        df = t.inverse_transform(df)
        out_cols = ["c1", "t1", "c2"]
        self.assertListEqual(list(df.columns), out_cols)

    def test_inverse_transform_keep_original_false_and_format(self):

        class C(r.RecipipeTransformer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def _transform(self, df):
                df = df[self.cols]
                df.columns = [self.col_format.format(i) for i in df.columns]
                return df
            def _inverse_transform(self, df):
                df = df[["c1_out", "c2_out"]]
                df.columns = ["c1", "c2"]
                return df
        t = C("c1", "c2", keep_original=False, col_format="{}_out")
        df = create_df_3dtypes()
        t.fit(df)
        df = t.transform(df)
        df = t.inverse_transform(df)
        out_cols = ["c1", "c2", "t1"]
        self.assertListEqual(list(df.columns), out_cols)

    def test_inverse_transform_keep_original_true_and_format(self):

        class C(r.RecipipeTransformer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def _transform(self, df):
                df = df[self.cols]
                df.columns = ["c1_out", "c2_out"]
                return df
            def _inverse_transform(self, df):
                df = df[["c1_out", "c2_out"]]
                df.columns = ["c1", "c2"]
                return df
        t = C("c*", keep_original=True, col_format="{}_out")
        df = create_df_3dtypes()
        t.fit(df)
        df = t.transform(df)
        df = t.inverse_transform(df)
        out_cols = ["c1", "c2", "t1"]
        self.assertListEqual(list(df.columns), out_cols)

    def test_inverse_transform_keep_original_without_original(self):

        class C(r.RecipipeTransformer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def _transform(self, df):
                df = df[self.cols]
                df.columns = ["c1_out", "c2_out"]
                return df
            def _inverse_transform(self, df):
                df = df[["c1_out", "c2_out"]]
                df.columns = ["c1", "c2"]
                return df
        t = C("c*", keep_original=True, col_format="{}_out")
        df = create_df_3dtypes()
        t.fit(df)
        df = t.transform(df)
        df = df.drop(["c1", "c2"], axis=1)
        df = t.inverse_transform(df)
        out_cols = ["c1", "c2", "t1"]
        self.assertListEqual(list(df.columns), out_cols)

    def test_transform_no_fit(self):
        """Raise exception if the transformer method is called without fit. """

        t = RecipipeTransformerMock()
        with self.assertRaises(ValueError):
            t.transform(None)

