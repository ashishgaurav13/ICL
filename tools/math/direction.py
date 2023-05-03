import numpy as np

class Direction2D:
    """
    2D direction with some properties and utilities.
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, value = None, mode = None):
        """
        Initialize Direction 2D.
        You can either specify mode (+/-x, +/-y) or an arbitrary decision
        using the value argument.
        """
        assert(mode in ['+x', '-x', '+y', '-y', None])
        assert(not(value != None and mode != None))
        self.mode = mode
        if value != None: self.value = value
        elif mode == '+x': self.value = np.array([1.0, 0.0])
        elif mode == '-x': self.value = np.array([-1.0, 0.0])
        elif mode == '+y': self.value = np.array([0.0, 1.0])
        elif mode == '-y': self.value = np.array([0.0, -1.0])
        else:
            print("Unknown direction?")
            exit(1)
        self.normalize()
    
    def normalize(self):
        """
        Make this direction be a 2D vector.
        """
        mag = np.sqrt(np.sum(np.square(self.value)))
        if mag > 0:
            self.value /= mag
    
    def transform_matrix(self, angle_rad):
        """
        Transformation matrix to rotate by `angle_rad` radians.
        """
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ])

    def angle(self):
        """
        Current angle. Measured clockwise.
        """
        x, y = self.value
        if x == 1 and y == 0: 
            return 0.0 # 2*np.pi - 0.0
        elif x == 0 and y == 1: 
            return 2*np.pi - np.pi/2
        elif x == -1 and y == 0: 
            return 2*np.pi - np.pi
        elif x == 0 and y == -1: 
            return 2*np.pi - 3*np.pi/2
        if x > 0 and y > 0:
            pre, t = 0.0, np.matmul(self.transform_matrix(0.0), self.value)
        elif x < 0 and y > 0:
            pre, t = np.pi/2, np.matmul(
                self.transform_matrix(- np.pi / 2), self.value)
        elif x < 0 and y < 0:
            pre, t = np.pi, np.matmul(
                self.transform_matrix(- np.pi), self.value)
        elif x > 0 and y < 0:
            pre, t = 3*np.pi/2, np.matmul(
                self.transform_matrix(- 3*np.pi / 2), self.value)
        return 2*np.pi - (pre + np.arctan(np.divide(t[1], t[0])))
    
    def dot(self, x):
        """
        Dot product with another Direction2D vector. This is a similarity measure.
        """
        assert(type(x) == Direction2D)
        return np.dot(self.value, x.value)