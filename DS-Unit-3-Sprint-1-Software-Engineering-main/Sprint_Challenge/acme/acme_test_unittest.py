#!/usr/bin/env python
import unittest
from acme import Product
from acme_report import generate_products, ADJECTIVES, NOUNS


class TestClass(unittest.TestCase):

    def test_default_product_price(self):
        """Test default product price being 10."""
        prod = Product('Test Product')
        self.assertEqual(prod.price, 10)

    def test_default_product_weight(self):
        """Test default product weight being 20."""
        prod = Product('Test Product')
        self.assertEqual(prod.weight, 20)

    def test_default_product_flammability(self):
        """Test default product flammability being 0.5"""
        prod = Product('Test Product')
        self.assertEqual(prod.flammability, 0.5)

    def test_methods(self):
        """Test default product flammability being 0.5"""
        prod = Product('Test Product')
        prod.stealability()
        self.assertEqual(prod.stealability(), 'Kinda stealable.')
        self.assertEqual(prod.explode(), '...boom!')

    def test_default_num_products(self):
        """
        This method determines if the generate products generate exactly 30 products
        :return: assertEqual
        """
        self.assertEqual(len(generate_products()), 30)

    def test_legal_names(self):
        """
        This method determines if all the legal names are used in generate_products()
        :return: assertListEqual
        """
        print('this is the last')
        self.unique_products = []
        for i in range(5):
            for j in range(5):
                # loop to make a list of products
                self.unique_products.append(ADJECTIVES[i] + " " + (NOUNS[j]))
        self.assertListEqual(self.unique_products, self.unique_products)


if __name__ == '__main__':
    unittest.main()
