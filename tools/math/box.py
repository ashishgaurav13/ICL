import numpy as np

class Box2D:
    """
    2D Box with some properties and utilities.
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, x1, x2, y1, y2, name):
        """
        Initialize Box2D, coordinates and center.
        """
        assert(x1 <= x2 and y1 <= y2)
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        self.name = name
        self.center = np.array([(self.x1+self.x2)/2, (self.y1+self.y2)/2])
    
    def __repr__(self):
        """
        String representation of the box.
        """
        return "%s <%.2f,%.2f,%.2f,%.2f>" % (self.name, self.x1, self.x2,
            self.y1, self.y2)
    
    def inside(self, x, y):
        """
        Is the point (x, y) inside the box?
        """
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def empty(self):
        """
        Is the box empty?
        """
        return self.x1 == self.x2 or self.y1 == self.y2

    def clip(self, x1, x2, y1, y2):
        """
        Clip the box to be within [x1, x2] and [y1, y2].
        """
        assert(x1 <= x2 and y1 <= y2)
        new_name = self.name+"_clipped"
        cx1, cx2, cy1, cy2 = self.x1, self.x2, self.y1, self.y2
        if x1 <= x2 <= cx1: cx1, cx2 = cx1, cx1
        elif cx2 <= x1 <= x2: cx1, cx2 = cx2, cx2
        else:
            if x1 >= cx1: cx1 = x1
            if x2 <= cx2: cx2 = x2
        if y1 <= y2 <= cy1: cy1, cy2 = cy1, cy1
        elif cy2 <= y1 <= y2: cy1, cy2 = cy2, cy2
        else:
            if y1 >= cy1: cy1 = y1
            if y2 <= cy2: cy2 = y2
        return Box2D(cx1, cx2, cy1, cy2, new_name)