import matplotlib.pyplot as plt
import tools
from inspect import isfunction
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from matplotlib.patches import Polygon
import numpy as np

class Plot2D:
    """
    Contains 2D plotting functions (done through matplotlib)
    """

    def __init__(self, l={}, cmds=[], mode="static", interval=None, 
        n=None, rows=1, cols=1, gif=None, legend=False, **kwargs):
        """
        Initialize a Plot2D.
        """
        assert(mode in ["static", "dynamic"])
        assert(type(l) == dict)
        assert(type(cmds) == list)
        self.legend = legend
        self.gif = gif
        self.fps = 60
        self.l = l
        self.mode = mode
        if interval is None: self.interval = 200 # 200ms
        else: self.interval = interval
        self.fig, self.ax = plt.subplots(nrows=rows, ncols=cols, **kwargs)
        self.empty = True
        self.cmds = cmds
        self.n = n
        self.reset()
        if self.mode == "dynamic":
            if n is None:
                self.anim = FuncAnimation(self.fig, self.step,
                    blit=False, interval = self.interval, repeat=False)
            else:
                self.anim = FuncAnimation(self.fig, self.step,
                    blit=False, interval = self.interval, frames=range(n+1),
                    repeat=False)
    
    def reset(self):
        """
        Reset and draw initial plots.
        """
        self.t = 0
        self.clear()
        self.data = {}
        self.objs = {}
        for key, val in self.l.items():
            self.data[key] = val(p=self, l=self.data, t=self.t)
        for i, cmd in enumerate(self.cmds):
            if type(cmd) == list:
                if cmd[0](p=self, l=self.data, t=self.t):
                    self.objs[i] = cmd[1](p=self, l=self.data, o=None, t=self.t)
            else:
                self.objs[i] = cmd(p=self, l=self.data, o=None, t=self.t)
        if type(self.ax) == np.ndarray:
            for item in self.ax:
                if type(item) == np.ndarray:
                    for item2 in item:
                        if self.legend:
                            item2.legend(loc='best')
                        item2.relim()
                        item2.autoscale_view()
                else:
                    if self.legend:
                        item.legend(loc='best')
                    item.relim()
                    item.autoscale_view()
        else:
            if self.legend:
                self.ax.legend(loc='best')
            self.ax.relim()
            self.ax.autoscale_view()

    def step(self, frame=None):
        """
        Increment the timer.
        """
        self.t += 1
        for key, val in self.l.items():
            self.data[key] = val(p=self, l=self.data, t=self.t)
        for i, cmd in enumerate(self.cmds):
            if type(cmd) == list:
                if cmd[0](p=self, l=self.data, t=self.t):
                    self.objs[i] = cmd[1](p=self, l=self.data, o=self.objs[i], t=self.t)
            else:
                self.objs[i] = cmd(p=self, l=self.data, o=self.objs[i], t=self.t)
        if type(self.ax) == np.ndarray:
            for item in self.ax:
                if type(item) == np.ndarray:
                    for item2 in item:
                        if self.legend:
                            item2.legend(loc='best')
                        item2.relim()
                        item2.autoscale_view()
                else:
                    if self.legend:
                        item.legend(loc='best')
                    item.relim()
                    item.autoscale_view()
        else:
            if self.legend:
                self.ax.legend(loc='best')
            self.ax.relim()
            self.ax.autoscale_view()

    def getax(self, loc=None):
        """
        Get the relevant axes object.
        """
        if loc is None:
            axobj = self.ax
        elif type(loc) == int or (type(loc) == list and len(loc) == 1):
            loc = int(loc)
            axobj = self.ax[loc]
        else:
            assert(len(loc) == 2)
            axobj = self.ax[loc[0], loc[1]]
        return axobj

    def imshow(self, X, loc=None, o=None, **kwargs):
        """
        Imshow X.
        """
        self.empty = False
        axobj = self.getax(loc=loc)
        if o is None:
            im = axobj.imshow(X, **kwargs)
            cbar = self.fig.colorbar(im, ax=axobj)
            return [im, cbar]
        else:
            im, cbar = o
            im.set_data(X)
            cbar.set_clim(np.min(X), np.max(X))
            return [im, cbar]
    
    def polygon(self, bbox, loc=None, o=None, **kwargs):
        """
        Draw polygon where bbox are the 4 X/Y coordinates of shape [4, 2].
        """
        self.empty = False
        axobj = self.getax(loc=loc)
        if o is not None:
            o.set_xy(bbox)
            return o
        else:
            poly = Polygon(bbox, **kwargs)
            axobj.add_patch(poly)
            return poly

    def line(self, X, Y, loc=None, o=None, **kwargs):
        """
        Line plot X/Y where `loc` are the subplot indices.
        """
        self.empty = False
        axobj = self.getax(loc=loc)
        if o is None:
            return axobj.plot(X, Y, **kwargs)[0]
        else:
            o.set_data(X, Y)
            return o
    
    def line_binary(self, X, Y, loc=None, o=None, trends=None, 
        trend_colors=['grey', 'pink'], **kwargs):
        """
        Line with two colors.
        """
        assert(trends != None)
        self.empty = False
        axobj = self.getax(loc=loc)
        ret = []
        n = len(X)
        lw = 3
        if n == 0:
            return None
        if o is not None and len(o) > 0:
            for oo in o:
                oo.remove()
        for i in range(n-1):
            ret += [axobj.plot(X[i:i+2], Y[i:i+2], color=trend_colors[0] \
                if trends[i] == "-" else trend_colors[1], linewidth=lw, **kwargs)[0]]            
        return ret
    
    def show(self, *args, **kwargs):
        """
        Show the entire plot in a nonblocking way.
        """
        if not self.empty:
            if not plt.get_fignums():
                # print("Figure closed!")
                return
            if hasattr(self, "shown") and self.shown == True:
                plt.draw()
                plt.pause(0.001)
                return
            if self.gif is None:
                plt.show(*args, **kwargs)
                self.shown = True
            else:
                assert(self.n is not None)
                plt.show(*args, **kwargs)
                self.shown = True
                self.anim.save(self.gif, writer=ImageMagickWriter(fps=self.fps, 
                    extra_args=['-loop', '1']),
                    progress_callback=lambda i, n: print("%d/%d" % (i, n)))
    
    def clear(self):
        """
        Clear the figure.
        """
        if type(self.ax) == np.ndarray:
            for item in self.ax:
                if type(item) == np.ndarray:
                    for item2 in item:
                        item2.cla()
                else:
                    item.cla()
        else:
            self.ax.cla()
        self.empty = True
    
    def candlestick(self, seq, loc=None, o=None, **kwargs):
        """
        Create a candlestick chart using Sequence.
        Sequence must have Index, Open, Close, Low, High.
        https://www.statology.org/matplotlib-python-candlestick-chart/
        """
        assert(seq.has_keys("Index", "Open", "Close", "Low", "High"))
        df = seq.pandas()
        if len(df.index) == 0:
            return [None, None, None, None]
        axobj = self.getax(loc=loc)
        width = (df.index.max()-df.index.min())/(1.5*len(df.index))
        width2 = width/3.0
        up = df[df["Close"]>=df["Open"]]
        down = df[df["Close"]<df["Open"]]
        col1 = (0,1,0,0.5)
        col2 = (1,0,0,0.5)
        if o is None:
            b1 = axobj.bar(up.index,up["Close"]-up["Open"],width,
                bottom=up["Open"],color=col1,**kwargs)
            b2 = axobj.bar(up.index,up["High"]-up["Low"],width2,
                bottom=up["Low"],color=col1,**kwargs)
            b3 = axobj.bar(down.index,down["Open"]-down["Close"],width,
                bottom=down["Close"],color=col2,**kwargs)
            b4 = axobj.bar(down.index,down["High"]-down["Low"],width2,
                bottom=down["Low"],color=col2,**kwargs)
            self.empty = False

            return [b1, b2, b3, b4]
        else:
            b1, b2, b3, b4 = o
            if [b1, b2, b3, b4] != [None, None, None, None]:
                b1.remove()
                b2.remove()
                b3.remove()
                b4.remove()
            b1 = axobj.bar(up.index,up["Close"]-up["Open"],width,
                bottom=up["Open"],color=col1,**kwargs)
            b2 = axobj.bar(up.index,up["High"]-up["Low"],width2,
                bottom=up["Low"],color=col1,**kwargs)
            b3 = axobj.bar(down.index,down["Open"]-down["Close"],width,
                bottom=down["Close"],color=col2,**kwargs)
            b4 = axobj.bar(down.index,down["High"]-down["Low"],width2,
                bottom=down["Low"],color=col2,**kwargs)
            self.empty = False

            return [b1, b2, b3, b4]