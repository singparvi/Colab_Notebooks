# Data Science Unit 3 Sprint Challenge 1

## Software Engineering - the Acme Way

In this sprint challenge you will write code and answer questions related to
object-oriented programming, code style/reviews, containers, and testing. You
may use any tools and references you wish, but your final code should reflect
*your* work and be saved in `.py` files (*not* notebooks), and (along with this
file including your written answers) uploaded to Canvas in the Sprint Challenge Submission form.

For all your code, you may only import/use the following:
- Other modules you write
- `pytest` (Install using pip)
- `random` (from the standard library)

As always, make sure to manage your time - get a section/question to "good
enough" and then move on to make sure you do everything. You can always revisit
and polish at the end if time allows.

This file is Markdown, so it can be helpful to render with VS Code or a
notebook to make it easier to look at.

Good luck!

### Part 1 - Keeping it Classy

As an employee of Acme Corporation, you're always looking for ways to better
organize the vast quantities and variety of goods your company manages and
sells. Everything Acme sells is considered a `Product`, and must have the
following fields (variables that live "inside" the class):

- `name` (string with no default)
- `price` (integer with default value 10)
- `weight` (integer with default value 20)
- `flammability` (float with default value 0.5)
- `identifier` (integer, automatically genererated as a random (uniform) number
  anywhere from 1000000 to 9999999, includisve)(inclusive).

Write a Python `class` to model the above data. Make sure you are *precise* in
your field names and types, and that your class has an `__init__` constructor
method with appropriate defaults (or lack thereof).

*Hint* - `random.randint` should be able to serve your random number needs.

Save the class in `acme.py`, and you can test your code in a Python repl as
follows:

```python
>>> from acme import Product
>>> prod = Product('A Cool Toy')
>>> prod.name
'A Cool Toy'
>>> prod.price
10
>>> prod.weight
20
>>> prod.flammability
0.5
>>> prod.identifier
2812086  # your value will vary
```

### Part 2 - Objects that Go!

The class you wrote in part 1 is nice, but it doesn't *do* anything - that is,
it doesn't have any *methods*. So let's add some! Specifically, add two methods:

- `stealability(self)` - calculates the price divided by the weight, and then
  returns a message: if the ratio is less than 0.5 return "Not so stealable...",
  if it is greater or equal to 0.5 but less than 1.0 return "Kinda stealable.",
  and otherwise return "Very stealable!"
- `explode(self)` - calculates the flammability times the weight, and then
  returns a message: if the product is less than 10 return "...fizzle.", if it is
  greater or equal to 10 but less than 50 return "...boom!", and otherwise
  return "...BABOOM!!"

Save your code, and you can test as follows:

```python
>>> from acme import Product
>>> prod = Product('A Cool Toy')
>>> prod.stealability()
'Kinda stealable.'
>>> prod.explode()
'...boom!'
```

### Part 3 - A Proper Inheritance

Of course, Acme doesn't just sell generic products - it sells all sorts of
special specific things!

Make a subclass of `Product` named `BoxingGlove` that does the following:

- Change the default `weight` to 10 (but leave other defaults unchanged)
- Override the `explode` method to always return "...it's a glove."
- Add a `punch` method that returns "That tickles." if the weight is below 5,
  "Hey that hurt!" if the weight is greater or equal to 5 but less than 15, and
  "OUCH!" otherwise
  
Example test run:

```python
>>> from acme import BoxingGlove
>>> glove = BoxingGlove('Punchy the Third')
>>> glove.price
10
>>> glove.weight
10
>>> glove.punch()
'Hey that hurt!'
>>> glove.explode()
"...it's a glove."
```

### Part 4 - Class Report

Now you can represent your inventory - let's use these classes and write an
`acme_report.py` module to generate random products and print a summary of them.
For the purposes of these functions we will only use the `Product` class.

Your module should include two functions:

- `generate_products()` should generate a given number of products (default
  30), randomly, and return them as a list
- `inventory_report()` takes a list of products, and prints a "nice" summary

For the purposes of generation, "random" means uniform - all possible values
should vary uniformly across the following possibilities:

- `name` should be a random adjective from `['Awesome', 'Shiny', 'Impressive',
  'Portable', 'Improved']` followed by a space and then a random noun from
  `['Anvil', 'Catapult' 'Disguise' 'Mousetrap', '???']`, e.g. `'Awesome Anvil'`
  and `Portable Catapult'` are both possible
- `price` and `weight` should both be from 5 to 100 (inclusive and independent,
  and remember - they're integers!)
- `flammability` should be from 0.0 to 2.5 (floats)

You should implement only depending on `random` from the standard library, your
`Product` class from `acme.py`, and built-in Python functionality.

For the report, you should calculate and print the following values:

- Number of unique product names in the product list
- Average (mean) price, weight, and flammability of listed products

Following is useful starting code for `acme_report.py`:

```python
#!/usr/bin/env python

from random import randint, sample, uniform
from acme import Product

# Useful to use with random.sample to generate names
ADJECTIVES = ['Awesome', 'Shiny', 'Impressive', 'Portable', 'Improved']
NOUNS = ['Anvil', 'Catapult', 'Disguise', 'Mousetrap', '???']


def generate_products(num_products=30):
    products = []
    # TODO - your code! Generate and add random products.
    return products


def inventory_report(products):
    pass  # TODO - your code! Loop over the products to calculate the report.


if __name__ == '__main__':
    inventory_report(generate_products())
```

The last lines let you test by running `python acme_report.py`. You should see
output like:

```
$ python acme_report.py 
ACME CORPORATION OFFICIAL INVENTORY REPORT
Unique product names: 19
Average price: 56.8
Average weight: 54.166666666666664
Average flammability: 1.258097155966675
```

It's OK for the specifics to vary (how you message/format), but it should output
and clearly identify all four relevant numbers.

*Hint* - a great way to figure out how many unique things you have is to use a
[set](https://docs.python.org/3.8/library/stdtypes.html#set-types-set-frozenset)
(a collection where all items are unique, and inserting an existing item just
doesn't do anything).
```
>>> s = set()
>>> s.add(1)
>>> s.add(2)
>>> s
{1, 2}
>>> s.add(3)
>>> s.add(1)
>>> s
{1, 2, 3}
```

### Part 5 - Measure twice, Test once

Make a file `acme_test.py` starting from the following code:

```python
#!/usr/bin/env python

import pytest
from acme import Product
from acme_report import generate_products, ADJECTIVES, NOUNS


def test_default_product_price(self):
    """Test default product price being 10."""
    prod = Product('Test Product')
    assert prod.price == 10
```
Complete the following:

- Add at least *2* more test methods to `acme_test.py` for the base
  `Product` class: at least 1 that tests default values (as shown), and one that
  builds an object with different values and ensures their `stealability()` and
  `explode()` methods function as they should
- Write at least 2 test functions that test the following:
  `test_default_num_products` which checks that it really does receive a list of
  length 30, and `test_legal_names` which checks that the generated names for a
  default batch of products are all valid possible names to generate (adjective,
  space, noun, from the lists of possible words)
  
*Hint* - `test_legal_names` is the trickiest of these, but may not be as bad as
you think. Consider using `in` as part of your `assert` statement, and remember that Python is
pretty handy at string processing. But if you get stuck, move on and revisit.

Note that `inventory_report()` is pretty tricky to test, because it doesn't
*return* anything - it just prints (a "side-effect"). For the purposes of this
challenge, don't worry about testing it - but as a stretch goal/something to
think about, it's a good ponderer.

### Part 6 - Style it Up

If you did the earlier parts in an editor that was linting your code (warning
you about violations of [PEP8 style](https://pep8.org/)) and you listened to it,
you're already done!

If not, go back and fix things! If you don't have a built-in tool for checking,
you can use [PEP8 online](http://pep8online.com/).

Go for lint-free! If there's a stubborn warning or two you can't fix though,
it's okay to leave a comment explaining it and move on.

### Part 7 - Questions (and your Answers)

Acme Corporation isn't just a few `.py` files. If you want to grow in your
career here, you'll have to answer the following:

- What, in your opinion, is an important part of code reviews? That is, what is
  something you pay attention to when you review code, and that you appreciate
  when others do the same for your code?
- We have an awful lot of computers here, and it gets pretty confusing with
  slightly different things running on all of them. How could containers help us
  improve this situation?

Answer both of these questions (baseline ~5 sentences) here in text.

Answer 1 - In my opinion the mort important part of code reviews is to point out the areas that may crash your app or make it behave abnormally. i.e if the coder has taken enough care to make sure the code behaves how he expects it to be. I like the code to be readable overall. A thorough testing is the most important aspect and the tests will only emulate the cases that you may have thought of. 

  I appreciate others telling me the ways I can improve my code, make it compact or more readable and if there are anything that may create an issue with my code tomorrow.

Answer 2 - Containers emulate an OS that is common as long as the same image is used. You can isolate the container from your local machine and make sure that is a code runs successfully with dependencies and other requirement, it will run smoothly on other containers with the same dependencies and requirements too.

### Part 8 - Turn it in!

Provide all the files you wrote (`acme.py`, `acme_report.py`, and
`acme_test.py`), as well as *this* file with your answers to part 7, in your **upload to Canvas**. 

Await feedback from Acme Corporation management. Thanks for your hard
work!

*Bonus!* Got this far? Read up on the [history of the fine Acme
Corporation](https://en.wikipedia.org/wiki/Acme_Corporation), with decades of
quality products and many satisfied customers (mostly coyotes).
