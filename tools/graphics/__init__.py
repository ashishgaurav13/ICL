"""
Drawables and graphics. Supports:

* 2D graphics using Pyglet/OpenGL
* 2D plotting using Matplotlib

Special classes:
* Car2D is a drawable as well as an environment.
"""

from .shapes import Rectangle2D
from .groups import *
from .car import Car2D
from .canvas import Canvas2D
from .plot import Plot2D