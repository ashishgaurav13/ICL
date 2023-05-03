import pyglet
from pyglet import gl
from tools.base import Drawable
from numpy import rad2deg
import os
from tools.utils import get_package_root_path
from tools.graphics import Rectangle2D

class Group(Drawable):
    """
    Collection of shapes, and possibly other collections.
    Taken from github.com/ashishgaurav13/wm2
    """
    
    def __init__(self, items = []):
        """
        Initialize a Group of drawables.
        """
        self.items = items
    
    def draw(self):
        """
        Individually draw items in this Group.
        """
        for item in self.items:
            item.draw()

class Text2D(Group):
    """
    2D Text label.
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, text, x, y, fontsize = 12, color = (0, 0, 0, 1),
        multiline = False, multiline_width = None):
        """
        Initialize a 2D text label.
        """
        
        color = [int(color[i]*255) for i in range(4)]
        if multiline and multiline_width == None: multiline_width = 300
        label = pyglet.text.Label(text, font_size = fontsize, color = color,
            x = x, y = y, multiline = multiline, width = multiline_width)
        super().__init__(items = [label])


class Image2D(Group):
    """
    Image which can be drawn on a 2D canvas. (needs a png)
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, url, x, y, w, h, rotation = 0, anchor_centered = False):
        """
        Initialize the image.
        """
        assert(url[:6] == "assets")
        assert(url[-3:] in ['png'])
        if url[-3:] == 'png': decoder = pyglet.image.codecs.png.PNGImageDecoder()
        url = os.path.join(get_package_root_path(), url)
        image = pyglet.image.load(url, decoder = decoder)
        if anchor_centered: 
            image.anchor_x, image.anchor_y = image.width//2, image.height//2
        scale_x, scale_y = w/image.width, h/image.height
        sprite = pyglet.sprite.Sprite(img = image)
        sprite.update(x = x, y = y, scale_x = scale_x, scale_y = scale_y,
            rotation = rotation)
        super().__init__(items = [sprite])

class StretchBackground2D(Group):
    """
    Background whose image is stretched.
    """

    def __init__(self, url, canvas):
        """
        Initialize StretchBackground2D.
        """
        w = canvas.width
        h = canvas.height
        bg = Image2D(url, 0, 0, w, h)
        super().__init__(items = [bg])

class Background2D(Group):
    """
    Infinitely repeating 2D background. Needs a Canvas2D.
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, url, canvas):
        """
        Initialize Background2D.
        """
        assert(url[:6] == "assets")
        assert(url[-3:] in ['png'])
        if url[-3:] == 'png': decoder = pyglet.image.codecs.png.PNGImageDecoder()
        url = os.path.join(get_package_root_path(), url)
        image = pyglet.image.load(url, decoder = decoder)
        self.bg = pyglet.image.TileableTexture.create_for_image(image)
        self.h, self.w = canvas.width, canvas.height
        super().__init__(items = [self.bg])
    
    def draw(self):
        """
        How to draw a Background2D.
        """
        self.bg.blit_tiled(0, 0, 0, self.h, self.w)


class GrassBackground2D(Group):
    """
    2D background using a grass image. Needs a Canvas2D.
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, canvas):
        """
        Initialize GrassBackground2D.
        """
        bg = Background2D('assets/driving/grass.png', canvas)
        super().__init__(items = [bg])


class Lane2D(Group):
    """
    2D rectangular lane with a gray image.
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, x1, x2, y1, y2):
        """
        Initialize Lane2D.
        """
        assert(x1 <= x2 and y1 <= y2)
        self.direction = 'x' if abs(y1-y2) < abs(x1-x2) else 'y'
        self.width = abs(y1-y2) if self.direction == 'x' else abs(x1-x2)
        self.length = abs(x1-x2) if self.direction == 'x' else abs(y1-y2)
        road_url = 'assets/driving/road.png'
        road = Image2D(road_url, x1, y1, x2-x1, y2-y1)
        super().__init__(items = [road])


class Intersection2D(Group):
    """
    2D rectangular intersection with a gray image.
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, x1, x2, y1, y2):
        """
        Initialize Intersection2D.
        """
        assert(x1 <= x2 and y1 <= y2)
        road_url = 'assets/driving/road.png'
        intersection = Image2D(road_url, x1, y1, x2-x1, y2-y1)
        super().__init__(items = [intersection])


class StopRegion2D(Group):
    """
    2D rectangular stop region.
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, x1, x2, y1, y2):
        """
        Initialize StopRegion2D.
        """
        assert(x1 <= x2 and y1 <= y2)
        region = Rectangle2D(x1, x2, y1, y2,
            color = (0.8, 0.8, 0.8, 1))
        super().__init__(items = [region])


class TwoLaneRoad2D(Group):
    """
    2D rectangular two lane road with a gray image.
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, x1, x2, y1, y2, sep):
        """
        Initialize TwoLaneRoad2D.
        """
        assert(x1 <= x2 and y1 <= y2)
        self.direction = 'x' if abs(y1-y2) < abs(x1-x2) else 'y'
        self.width = abs(y1-y2) / 2 if self.direction == 'x' else abs(x1-x2) / 2
        self.length = abs(x1-x2) if self.direction == 'x' else abs(y1-y2)
        if self.direction == 'x':
            super().__init__(items = [
                Lane2D(x1, x2, y1, y1+self.width),
                Lane2D(x1, x2, y1+self.width, y2),
                Rectangle2D(x1, x2, y1+self.width-sep/2, y1+self.width+sep/2,
                    color = (1, 1, 1, 1)),
            ])
        else:
            super().__init__(items = [
                Lane2D(x1, x1+self.width, y1, y2),
                Lane2D(x1+self.width, x2, y1, y2),
                Rectangle2D(x1+self.width-sep/2, x1+self.width+sep/2, y1, y2,
                    color = (1, 1, 1, 1)),
            ])