def someFunction(**kwargs):
    print(kwargs['a'])
    try:
        print(kwargs['b'])
    except:
        print('value of b not given')
    finally:
        print('function execution finished')

# when number of arguements not fixed given as tuple
def someOtherFunction(*names):
    print(names)

# someFunction(a='something')
someOtherFunction((2,)) 

# Topics also covered - ref https://www.freecodecamp.org/news/the-python-handbook/#classesinpython
# Annotations
def increment(n: int) -> int:
   return n + 1

# Decorator


def logtime(func):
    def wrapper():
        # do something before
        val = func()
        # do something after
        return val
    return wrapper

@logtime
def hello():
    print('hello!')
    
# DocString

class Dog:
    """A class representing a dog"""
    def __init__(self, name, age):
        """Initialize a new dog"""
        self.name = name
        self.age = age

    def bark(self):
        """Let the dog bark"""
        print('WOF!')