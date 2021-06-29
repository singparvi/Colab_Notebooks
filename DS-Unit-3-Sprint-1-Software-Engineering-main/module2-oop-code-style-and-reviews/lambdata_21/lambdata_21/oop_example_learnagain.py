"""OOP example for module 2"""



import pandas as pd
import numpy as np

class BareMinimumClass:
    pass

class Complex:
    def __init__(self, realpart, imagpart):
        """
        Constructor for complex numbers
        Complex numbers have a real part and an imag part
        """
        self.r = realpart
        self.i = imagpart


    def add(self, other_complex):
        """"""
        self.r += other_complex.r
        self.i += other_complex.i

        def __repr__(self):
            return '({}, {})'.format(self.r, self.i)

num1 = Complex(3,5)
num2 = Complex(4,2)

num1.add(num2)

print(num1.r, num1.i)