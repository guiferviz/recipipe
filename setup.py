
from os import listdir
from os.path import join

from setuptools import setup, find_packages
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements


PACKAGE_NAME = "recipipe"
DESCRIPTION = "Improved pipelines for data science projects."
KEYWORDS = "datascience machinelearning pipeline sklearn pandas"
AUTHOR = "guiferviz"
LICENSE = "MIT, Copyright " + AUTHOR


# Creates a __version__ variable.
with open(PACKAGE_NAME + "/_version.py") as file:
    exec(file.read())

# Read requirements.
req = parse_requirements("requirements.txt", session="hack")
REQUIREMENTS = []
for i in req:
    if getattr(i, "req", None) is not None:
        REQUIREMENTS.append(str(i.req))
    elif getattr(i, "requirement", None) is not None:  # pip >= 20.1
        REQUIREMENTS.append(str(i.requirement))
    else:
        print("ERROR: I don't understand this requirement...")
print("Requirements:", REQUIREMENTS)

# Install all packages in the current dir except tests.
PACKAGES = find_packages(exclude=["tests", "tests.*"])
print("Packages:", PACKAGES)

setup(name=PACKAGE_NAME,
      version=__version__,
      description=DESCRIPTION,
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      keywords=KEYWORDS,
      author=AUTHOR,
      license=LICENSE,
      packages=PACKAGES,
      install_requires=REQUIREMENTS,
      entry_points={
          "console_scripts": [
              "{} = {}.__main__:main".format(PACKAGE_NAME, PACKAGE_NAME)
          ]
      },
      zip_safe=False)

