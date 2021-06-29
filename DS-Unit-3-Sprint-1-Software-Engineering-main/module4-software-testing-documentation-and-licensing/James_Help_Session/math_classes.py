class SimpleMath:
  """A class for simple math functions."""

  def __init__(self, a, b):
    self.a = a
    self.b = b

  def add(self):
    """Adds two numbers."""
    return self.a + self.b

  def subtract(self):
    """Subtracts two numbers."""
    return self.a - self.b

  def multiply(self):
    """Multiplies two numbers."""
    return self.a * self.b

  def divide(self):
    """Divides two numbers."""
    return self.a // self.b



class Complex(SimpleMath):
  """Performs more complex calculations!"""

  def __init__(self, a, b):
    super().__init__(a, b)

  def exponent(self):
    """Returns 'a' to the 'b'."""
    return self.a ** self.b

  def nth_root(self):
    """Returns 'b'th root of a."""
    return self.a ** (1 / float(self.b))
