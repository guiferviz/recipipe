
<p align="center">
  <img src="https://raw.githubusercontent.com/guiferviz/recipipe/master/doc/_static/logo/logo.png"
       alt="Recipipe logo. A muffing with a couple of pipes over a green background." />
</p>

![Minimun Python version >= 3.6](https://img.shields.io/badge/Python-%3E=3.6-blue?style=flat&logo=python)
[![PyPI version](https://badge.fury.io/py/recipipe.svg)](https://badge.fury.io/py/recipipe)
[![Python tests](https://github.com/guiferviz/recipipe/workflows/Python%20tests/badge.svg)](https://github.com/guiferviz/recipipe/actions?query=workflow%3A%22Python+tests%22)
[![Coverage status](https://coveralls.io/repos/github/guiferviz/recipipe/badge.svg?branch=master)](https://coveralls.io/github/guiferviz/recipipe?branch=master)
[![Build docs](https://github.com/guiferviz/recipipe/workflows/Build%20Docs/badge.svg)](https://guiferviz.com/recipipe/)
[![GitHub license](https://img.shields.io/github/license/guiferviz/recipipe.svg)](https://github.com/guiferviz/recipipe/blob/master/LICENSE)

[![Twitter @guiferviz](https://img.shields.io/twitter/follow/guiferviz?style=social)](https://twitter.com/guiferviz)

Improved pipelines for data science projects.


# Getting started


## Install from PyPI

    pip install recipipe

All the dependencies will be installed automatically.


## Install from source

Clone the repository and run:

	pip install .

Install the package in a dev environment with:

    pip install -e .

All the dependencies will be installed automatically.


## Tutorials and examples

Learn how to use Recipipe analyzing data from weird creatures from another planet:
[Recipipe getting started tutorial](examples/paranoids.ipynb).


## Running the tests

Run all the test using:

    pytest

Run an specific test file with:

    pytest tests/<filename>

Run tests with coverage using:

    coverage run --source=recipipe -m pytest


# What's the meaning of Recipipe?

It comes from a beautiful R library called [recipes][recipesR] and the concept
of [pipelines][pipelinesWikipedia].

    recipes + pipelines = recipipe

That explains the logo of a muffing (recipes) holding some pipes (pipelines).

<p align="center">
  <img src="https://raw.githubusercontent.com/guiferviz/recipipe/master/doc/_static/logo/logo.png"
       alt="Recipipe logo. A muffing with a couple of pipes over a green background."
       width=100 />
</p>


# License

This project is licensed under the **MIT License**, see the
[LICENSE][license] file for details.


# Author

*guiferviz*, contributions are more than welcome.

[![Twitter @guiferviz](https://img.shields.io/twitter/follow/guiferviz?style=social)](https://twitter.com/guiferviz)


[license]: https://github.com/guiferviz/recipipe/blob/master/LICENSE
[recipesR]: https://github.com/tidymodels/recipes
[pipelinesWikipedia]: https://en.wikipedia.org/wiki/Pipeline_(computing)

