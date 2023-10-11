class Animal:
    def walks(self):
        print('Walking')

class Dog(Animal):
    def speak(self):
        print('Woof')


class Cat(Animal):
    def __init__(self,name) -> None:
        super().__init__()
        self.name=name
    def speak(self):
        print(f'Meow{self.name}')


cat =  Cat('cat')
cat.speak()
cat.walks()