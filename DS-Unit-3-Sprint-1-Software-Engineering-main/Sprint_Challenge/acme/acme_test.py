#!/usr/bin/env python

import pytest
from acme import Product
from acme_report import generate_products, ADJECTIVES, NOUNS


def test_default_product_price(self):
    """Test default product price being 10."""
    prod = Product('Test Product')
    assert prod.price == 10


def test_default_product_weight(self):
    """Test default product weight being 20."""
    prod = Product('Test Product')
    assert prod.weight == 20


def test_default_product_flammability(self):
    """Test default product flammability being 0.5"""
    prod = Product('Test Product')
    assert prod.flammability == 0.5


def test_methods(self):
    """Test default product flammability being 0.5"""
    prod = Product('Test Product')
    prod.stealability()
    assert (prod.stealability() == 'Kinda stealable.')
    assert (prod.explode() == '...boom!')


def test_default_num_products(self):
    """
    This method determines if the generate products generate
    exactly 30 products
    :return: assertEqual
    """
    assert len(generate_products()) == 30


def test_legal_names(self):
    """
    This method determines if all the legal names are used in
    generate_products()
    :return: assertListEqual
    """
    unique_products = []
    for i in range(5):
        for j in range(5):
            # loop to make a list of products
            unique_products.append(ADJECTIVES[i] + " " + (NOUNS[j]))
    assert (unique_products in generate_products())


if __name__ == '__main__':
    pytest.main()
