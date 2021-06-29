import unittest

from math_classes import SimpleMath, Complex

s = SimpleMath(4, 2)
c = Complex(64, 3)


# Perform basic assert statements for Pytest
# Stick to the schema of naming them 'test_*'

def test_add():
    assert (s.add() == 6)


def test_subtract():
    assert (s.subtract() == 2)


def test_multiply():
    assert (s.multiply() == 8)


def test_divide():
    assert (s.divide() == 2)


# You can also do testing with the Unittest package

class TestComplex(unittest.TestCase):

    def test_exponent(self):
        self.assertEqual(c.exponent(), 262144)  # assert c.exponent() == 262144
        self.assertIsInstance(c.exponent(), int)  # assert type(262144) == int

    def test_nth_root(self):
        self.assertAlmostEqual(c.nth_root(), 4)  # assert 3.99999... ~= 4
        self.assertIsInstance(c.nth_root(), float)  # assert type(3.99999...) == float

# The code below will only run tests that are set up as classes (the above)
# if __name__ == '__main__':
#   unittest.main()
