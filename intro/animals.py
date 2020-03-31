"""
Representations of animals.
"""

class Animal(object):
    """
    An animal with a given `name` and `sound`.
    """
    def __init__(self, name, sound='...'):
        self.name = name
        self.sound = sound

    def talk(self):
        print(self.name + ': ' + self.sound)
        
    def shout(self):
        print(self.name + ': ' + self.sound.upper())

    def __add__(self, other):
        return Animal(self.name + other.name,
                     self.sound + other.sound)


class Duck(Animal):
    """
    A duck with a given `name`.
    """
    def __init__(self, name):
        super().__init__(name, 'quack')


class Dog(Animal):
    """
    A dog with a given `name`.
    """
    def __init__(self, name):
        super().__init__(name, 'woof')
