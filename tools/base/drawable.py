import abc

class Drawable(abc.ABC):
    """
    Anything that can be drawn.
    """

    @abc.abstractmethod
    def draw(self):
        """
        How to draw this object.
        """
        pass