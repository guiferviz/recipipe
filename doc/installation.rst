
Installation
============

From PyPi
---------

Using *pip*::

    pip install recipipe


From source code
----------------

The latest code is in the master branch of the GitHub repository
(https://github.com/guiferviz/recipipe).
Clone the repository and change the current directory to the root directory of
the project:

.. code-block:: bash

    git clone git@github.com:guiferviz/recipipe.git
    cd recipipe

As always, consider installing the package in a virtual environment.
Install *recipipe* using *pip*:

.. code-block:: bash

    pip install .

All the dependencies will be installed automatically.

If you are developing you need to install some extra dependencies (for running
tests and generating docs):

.. code-block:: bash

    pip install -r requirements_dev.in

The installation is not really needed for running tests.
If you want to install it, it's recommended to install it using the `-e`/
`--editable` mode (i.e. setuptools "develop mode"):

.. code-block:: bash

    pip install -e .

If you have any problems with the dependencies you can always use the exacts
versions of the packages that are know to work well.
Those dependencies are in `requirements.txt` and `requirements_dev.txt` for
production and development dependencies respectively.
Install them using:

.. code-block:: bash

    pip install -r <requirements-filename>

