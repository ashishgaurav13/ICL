from functools import total_ordering
import numpy as np

@total_ordering
class Feature:
    """
    Holds a certain feature.
    """

    def __init__(self, name, value):
        """
        Create the feature.
        """
        self.name = name
        self.value = value
        self.dtype = type(value)
        assert(type(value) in [float, int, bool])
   
    def __repr__(self):
        """
        String representation of Feature.
        """
        return "%s:%g" % (self.name, self.value)
    
    def __eq__(self, v):
        """
        Equality comparison operator.
        """
        if type(v) in [float, int, bool]:
            return self.value == v
        elif type(v) == Feature:
            return self.value == v.value
        elif type(v) == str:
            return str(self) == v
        else:
            return NotImplementedError()
    
    def __lt__(self, v):
        """
        Less than comparison operator.
        """
        if type(v) in [float, int, bool]:
            return self.value < v
        elif type(v) == Feature:
            return self.value < v.value
        else:
            return NotImplementedError()

    def numpy(self, dtype = np.float32):
        """
        Convert to numpy format.
        """
        return dtype(self.value)

class Features:
    """
    Holds a bunch of features, almost `tools.data.Bits`
    but the features can be any value.
    """

    def __init__(self, f):
        """
        Initialize Features.
        """
        assert(type(f) == dict)
        self.o = {key: Feature(key, value) for key, value in f.items()}
    
    def __getitem__(self, index):
        """
        Get feature value by index.
        """
        return self.o[index].value
    
    def __setitem__(self, index, value):
        """
        Set feature value by index.
        """
        if index not in self.o:
            self.o[index] = Feature(index, value)
        else:
            self.o[index].value = value

    def __repr__(self):
        """
        String representation of features.
        """
        return ", ".join([str(item) for item in self.o.values()])
    
    def keys(self):
        """
        Get feature keys.
        """
        return self.o.keys()
    
    def numpy(self, dtype = np.float32):
        """
        Convert to numpy format.
        """
        return np.array([item.numpy(dtype) for item in self.o.values()])
    
    def get_dict(self):
        """
        Convert to dictionary.
        """
        return {key: self.o[key].value for key in self.o.keys()}