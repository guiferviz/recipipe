
import glob
import json
import os
import re

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

import pytest

import unittest


ROOT_DIR = os.path.abspath(".")
NOTEBOOKS = glob.glob("./examples/*.ipynb")

RE_UUID = re.compile(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', re.I)


def read_notebook(filename):
    nb = None
    with open(filename) as f:
        nb = nbformat.read(f, as_version=4)
    return nb


def clean_output(o):
    # Transform object to text using always the same order to detect changes.
    s = json.dumps(o, sort_keys=True)
    # Avoid error: bad escape \u at position X.
    s = s.replace(r"\u", "")
    # Remove UUIDs. Useful for generated HTML.
    s = RE_UUID.sub("", s)
    return s


# One test per each example notebook.
@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook(notebook):
    # Read input notebook.
    os.chdir(ROOT_DIR)
    nb = read_notebook(notebook)
    nb_original = read_notebook(notebook)  # Idk how to deep copy a notebook
    # Execute nb.
    os.chdir("./examples/")
    ep = ExecutePreprocessor(timeout=60, kernel_name='python3')  # 60s
    ep.preprocess(nb)
    # Compare output cells.
    i = 0
    for co, c in zip(nb_original.cells, nb.cells):
        if co.cell_type == c.cell_type == "code":
            i += 1
            co = clean_output(co.outputs)
            c = clean_output(c.outputs)
            unittest.TestCase().assertEqual(co, c)

