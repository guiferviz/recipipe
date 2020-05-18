
from recipipe import greet
from recipipe import __author__
from recipipe import __version__


def main():
    """Main entry point.

    This function is called when you execute the module
    (for example using `recipipe` or `python -m recipipe`).
    At the moment, nothing interesting is done here.
    """
    greet()
    print()
    print(f"Recipipe, {__version__}")
    print(f"Made with love by {__author__}")


if __name__ == "__main__":  # pragma: no cover
    main()

