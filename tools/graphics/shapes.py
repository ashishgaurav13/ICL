import pyglet
from pyglet import gl
from tools.base import Drawable

class Rectangle2D(Drawable):
    """
    Basic 2D rectangle using Pyglet.
    Taken from github.com/ashishgaurav13/wm2
    """
    
    def __init__(self, x1, x2, y1, y2, color = (0, 0, 0, 1)):
        """
        Initialize a rectangle coordinates and color.
        """
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.color = color
    
    def draw(self):
        """
        Draw rectangle using Pyglet/OpenGL.
        """
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(*self.color)
        gl.glVertex3f(self.x1, self.y1, 0)
        gl.glVertex3f(self.x1, self.y2, 0)
        gl.glVertex3f(self.x2, self.y2, 0)
        gl.glVertex3f(self.x2, self.y1, 0)
        gl.glEnd()
