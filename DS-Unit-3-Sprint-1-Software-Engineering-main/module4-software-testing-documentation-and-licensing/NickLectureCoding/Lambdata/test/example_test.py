"""
Example tests using standard documentation.
https://docs.python.org/3/library/unittest.html
"""

import unittest


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_in(self):
        s = 'This is a nice looking string'
        self.assertTrue('T' in s)
        self.assertFalse('X' in s)
        self.assertIn('T', s)

if __name__ == '__main__':
    unittest.main()