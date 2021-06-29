# import necassary packages

import random


class Product:
    """
    Product class that takes name (mandatory), price (default=10),
    weight (default=20), flammability (default=0.5)
    """

    def __init__(self, name, price=10, weight=20, flammability=0.5):
        """
        This method is to initialize the class Product by taking in
        name, price, weight and flammability.
        """
        self.name = str(name)
        self.price = int(price)
        self.weight = int(weight)
        self.flammability = float(flammability)
        self.identifier = random.randint(1000000, 9999999)

    def stealability(self):
        """
        This method determines if a product is stealable or not.
        calculates the price divided by the weight, and
        then returns a message: if the ratio is less than 0.5 return
        "Not so stealable...", if it is greater or
        equal to 0.5 but less than 1.0 return "Kinda stealable.",
        and otherwise return "Very stealable!"
        """
        if (self.price / self.weight) < 0.5:
            return "Not so stealable..."

        elif ((self.price / self.weight) >= 0.5) and ((self.price
                                                       / self.weight) < 1.0):
            return "Kinda stealable."

        else:
            return "Very stealable!"

    def explode(self):
        """
        This method determines if a product is flammability or not.
        calculates the flammability times the weight, and
        then returns a message: if the product is less than 10 return
        "...fizzle.", if it is greater or equal to 10 but
        less than 50 return "...boom!", and otherwise return "...BABOOM!!"
        """
        if (self.flammability * self.weight) < 10:
            return "...fizzle."

        elif ((self.flammability * self.weight) >= 10) and (
            (self.flammability * self.weight) < 50
        ):
            return "...boom!"

        else:
            return "...BABOOM!!"


class BoxingGlove(Product):
    """
    Subclass BoxingGlove that inherits from Product class
    """

    def __init__(self, name, price=10, weight=10, flammability=0.5):
        """
        Signature to instantiate the class is the same as class Product
        i.e. pass name (mandatory) and other parameters
        are optional
        :param name: string
        :param price: default 10
        :param weight: default 10
        :param flammability: default 0.5
        """
        super().__init__(name, price, weight, flammability)

    def explode(self):
        """
        This method to always return "...it's is a glove"
        """
        return "...it's a glove"

    def punch(self):
        """
        This method returns "That tickles." if the weight is below 5,
        "Hey that hurt!" if the weight is greater or equal
        to 5 but less than 15, and "OUCH!" otherwise
        """
        if self.weight < 5:
            return "That tickles."

        elif (self.weight >= 5) and (self.weight < 15):
            return "Hey that hurt!"

        else:
            return "OUCH!"
