class Monkey:
    def __init__(self, name, price, keybind) -> None:
        self.name = name
        self.price = price
        self.keybind = keybind



monkey_dict = {
    'ninja': Monkey("ninja", 340, 'd'),
    'bomb': Monkey("bomb", 445, 'e')
}
