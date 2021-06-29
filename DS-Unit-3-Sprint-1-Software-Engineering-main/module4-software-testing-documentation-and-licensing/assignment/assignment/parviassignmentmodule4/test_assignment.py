import unittest
# import pandas as pd
from helper_functions_2 import CustomDF, CustomDF2


class TestHelperFunctions(unittest.TestCase):
    """This is to test the function for Custom DF and CustomDF2"""

    def setUp(self):
        """This is to set up the parameters to run the test successfully"""
        self.DF = CustomDF(
            '/Users/rob/G_Drive_sing.parvi/Colab_Notebooks/Unit-3-Sprint-1-Software-Engineering-main/module4-software-testing-documentation-and-licensing/assignment/assignment/Dates.csv')
        self.DF2 = CustomDF2(
            '/Users/rob/G_Drive_sing.parvi/Colab_Notebooks/Unit-3-Sprint-1-Software-Engineering-main/module4-software-testing-documentation-and-licensing/assignment/assignment/Dates.csv',
            'Date')

    def test_null_count(self):
        """This is to check the type of the DataFrame"""
        self.assertEqual(self.DF.nullcount(), 0)

    def test_split_dates(self):
        """This is to check the that the year value in DF2 is 2001"""
        self.assertEqual(self.DF2.split_dates().year[0], 2001)


if __name__ == '__main__':
    unittest.main()
