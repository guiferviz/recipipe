
from unittest import TestCase

from recipipe.__main__ import main


class MainTest(TestCase):

    def test_no_error_main(self):
        main()

