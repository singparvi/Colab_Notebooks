# Objective 01 - understand and follow Python namespaces and imports


def fib(n):
    a, b = 0, 1
    while a<n:
        print (a, end=' ')
        a, b = b, a+b


def fib2(n):
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a+b
    return20 result


print('BBB')

# Objective 02 - create a Python package and install dependencies in a dedicated environment
