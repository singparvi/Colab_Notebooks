#!/usr/bin/env python

from random import randint, sample, uniform
from acme import Product

# Useful to use with random.sample to generate names
ADJECTIVES = ["Awesome", "Shiny", "Impressive", "Portable", "Improved"]
NOUNS = ["Anvil", "Catapult", "Disguise", "Mousetrap", "???"]


# make the functions
def generate_products(num_products=30):
    """
    This function helps to generate products based on the num_products
    argument passed
    :param num_products: default 30
    :return: products
    """
    products = []
    # TODO - your code! Generate and add random products.
    # loop to
    for i in range(num_products):
        # loop to make a list of products
        products.append(
            sample(ADJECTIVES, 5)[randint(0, 4)] + " " +
            sample(NOUNS, 5)[randint(0, 4)]
        )
    return products


def inventory_report(products):
    """
    This function helps to take in the products as a list and return a
    summary report of the inventory.
    :param products: products as a list
    :return: summary of inventory report
    """
    print("ACME CORPORATION OFFICIAL INVENTORY REPORT")
    print("Unique product names: ", len(set(products)))
    # initialize the variables used
    products_class = list(range(30))
    total_price = 0
    total_weight = 0
    total_flammability = 0
    for i in range(len(products)):
        # loop to make a list of class with randomized parameters and
        # do the other calculations
        products_class[i] = Product(
            products[i],
            price=randint(5, 100),
            weight=randint(5, 100),
            flammability=uniform(0, 2.5),
        )
        total_price += products_class[i].price
        total_weight += products_class[i].weight
        total_flammability += products_class[i].flammability

    # calculate
    average_price = total_price / len(products)
    average_weight = total_weight / len(products)
    average_flammability = total_flammability / len(products)

    # print
    print("Average price: ", average_price)
    print("Average weight: ", average_weight)
    print("Average flammability: ", average_flammability)
    pass  # TODO - your code! Loop over the products to calculate the
    # report.


if __name__ == "__main__":
    inventory_report(generate_products())
