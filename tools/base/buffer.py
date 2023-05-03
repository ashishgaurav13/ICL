import abc

class Buffer(abc.ABC):
    """
    Stores experiences or data.
    """

    @abc.abstractmethod
    def __len__(self):
        """
        Return current size of data.
        """
        pass

    @abc.abstractmethod
    def add(self, data):
        """
        Add `data` to the buffer.
        """
        pass

    @abc.abstractmethod
    def sample(self, n):
        """
        Sample n samples from stored data.
        """
        pass