"""This module allows you to test the library without installing it.

Write `from .context import recipipe` in any test file to import the
library.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import recipipe
