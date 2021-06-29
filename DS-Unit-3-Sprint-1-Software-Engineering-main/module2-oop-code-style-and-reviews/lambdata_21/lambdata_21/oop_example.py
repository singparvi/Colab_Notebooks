""" OOP example for module 2"""

import pandas as pd
import numpy as np

class MyDataFrame(pd.DataFrame):
    pass

class BareMinimumClass: # capitalize each layer
    pass

class Complex:
    def __init__(self, realpart, imagpart):    # making a constructor, this is the first thing that gets executed when instantiating this class
        # constructor set the values of attributes that we like to use. This gets executed automatically.
        # every constructor has a keyword self argument. Instance of the class itself.
        """Constructor for complex numbers.
        Complex numbers have a real part and an imaginary part
        """
        self.r = realpart # object coming from class will have now has realpart as a variable. Setting attribute using realpart
        self.i = imagpart # object coming from class will not have a imagpart as a variable

    def add(self, other_complex):
        self.r += other_complex.r
        self.i += other_complex.i

    def __repr__(self): # representation
        """This basically returns a string when we print and echo location.
        Instead we will get the string we are going to specify"""

        return f'({self.r}, {self.i})'


class SocialMediaUser:
    def __init__(self, name, location, upvotes=0):
        self.name = str(name)
        self.location = location
        self.upvotes = int(upvotes)

    def receive_upvotes(self, num_upvotes=1):
        self.upvotes += num_upvotes

    def is_popular(self):
        return self.upvotes > 100

class Animal:
    """
    General Representation of Animals
    """
    def __init__(self, name, weight, diet_type):
        self.name = str(name)
        self.weight = float(weight)
        self.diet_type = diet_type

    def run(self):
        return 'Vroom, Vroom, I go quick'

    def eat(self, food):
        return 'Huge fan of that '+str(food)

    # inheritance

class Sloth(Animal): # inheriting Animal in Sloth
    def __init__(self, name, weight, diet_type, num_namps):
        # self.name = str(name)
        # self.weight = float(weight)
        # self.diet_type = diet_type
        super().__init__(name, weight, diet_type) # super get all from parent class
        self.num_namps = int(num_namps)

    # new function specific to this function
    def say_something(self):
        return 'This is a sloth of typing'

    #override previous class method from parent class
    def run(self):
        return 'I am a slow sloth guy'


class Parvi:
    def __init__(self, name):
        self.name = name


if __name__ == '__main__': # this means the code will run only if this py is executed.Also first thing executed when in python module
    num1 = Complex(3,5) # real part 3 and imag 5
    num2 = Complex(4,2)
    num1.add(num2)
    print(num1.r, num1.i)

    # instantiate a user
    user1 = SocialMediaUser('Justin', 'Provo')
    user2 = SocialMediaUser('Nick', 'Logan', 200)
    user3 = SocialMediaUser(name='Carl', location='Costa Rica', upvotes=100000)
    user4 = SocialMediaUser(name='George Washington', location='Djibouti', upvotes=2)

    print(f'name: {user4.name}, is popular: {user4.is_popular()}, num upvotes: {user4.upvotes}')
    print(f'name: {user3.name}, is popular: {user3.is_popular()}, num upvotes: {user3.upvotes}')





