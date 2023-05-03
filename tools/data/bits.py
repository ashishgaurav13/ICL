class Bits:
    """
    A bit-control class that allows us bit-wise manipulation as shown in the
    example:

    bits = Bits()
    bits[0] = False
    bits[2] = bits[0]

    Slicing is not allowed.

    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, value=0):
        self._d = value

    def __getitem__(self, index):
        return (self._d >> index) & 1

    def __setitem__(self, index, value):
        value = bool(value) # ADDED
        value = (value & 1) << index
        mask = 1 << index
        self._d = (self._d & ~mask) | value

    def __int__(self):
        return self._d