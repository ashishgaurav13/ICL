import glob, os
from tools.utils import get_package_root_path
import bs4
import matplotlib.pyplot as plt
import utm
import pandas
from matplotlib.patches import Polygon
import numpy as np
import matplotlib
import copy

# Reading XML file: 
# https://www.geeksforgeeks.org/reading-and-writing-xml-files-in-python/
# pip install lxml
# pip install beautifulsoup4

# Lat Lon to UTM coordinates
# pip install utm

class Node:

    def __init__(self, nodetag, utmx, utmy):
        for k, v in nodetag.attrs.items():
            setattr(self, k, v)
        self.x, self.y, _, _ = utm.from_latlon(float(self.lat), float(self.lon))
        self.x -= utmx
        self.y -= utmy
        # self.x /= (0.10106*4)
        # self.y /= (0.10106*4)

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.x, self.y, color="blue",
            marker="o", s=5, linewidths=0)

class Way:

    def __init__(self, waytag, nodes):
        for k, v in waytag.attrs.items():
            setattr(self, k, v)
        self.data = []
        for child in waytag.children:
            if child.name == "nd":
                for node in nodes:
                    if node.id == child.get("ref"):
                        self.data += [node]
                        break
            elif child.name == "tag":
                setattr(self, child.get("k"), child.get("v"))
    
    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        xs, ys = [], []
        for node in self.data:
            xs += [node.x]
            ys += [node.y]
        ax.plot(xs, ys, color='red')

class Relation:

    def __init__(self, reltag, ways):
        for k, v in reltag.attrs.items():
            setattr(self, k, v)
        self.data = []
        self.roles = []
        for child in reltag.children:
            if child.name == "member":
                for way in ways:
                    if way.id == child.get("ref"):
                        self.data += [way]
                        self.roles += [child.get("role")]
                        break
            elif child.name == "tag":
                setattr(self, child.get("k"), child.get("v"))
        self.mid = []
        for node in self.data[0].data:
            closest = None
            closestdist = float('inf')
            for node2 in self.data[1].data[::-1]: # find closest
                dist = ((node.x-node2.x)**2+(node.y-node2.y)**2)**0.5
                if dist < closestdist:
                    closest = node2
                    closestdist = dist
            anothernode = copy.deepcopy(node)
            anothernode.x = (node.x+closest.x)/2
            anothernode.y = (node.y+closest.y)/2
            self.mid += [anothernode]

    def plot(self, color = None, alpha = None, ax=None):
        assert(len(self.data) == 2) # left/right roles only
        polydata = self.data[0].data + self.data[1].data[::-1]
        xs, ys = [], []
        for node in polydata:
            xs += [node.x]
            ys += [node.y]
        vert = np.array([xs, ys]).T
        if color is None:
            polygon = Polygon(vert, True, facecolor=(np.random.randint(256)/255.,
                np.random.randint(256)/255., np.random.randint(256)/255.), alpha=0.4)
        else:
            if alpha is None:
                alpha = 0.4
            polygon = Polygon(vert, True, facecolor=color, alpha=alpha)
        if ax is None:
            ax = plt.gca()
        ax.add_patch(polygon)
        xs, ys = [], []
        for node in self.mid:
            xs += [node.x]
            ys += [node.y]
        ax.plot(xs, ys, color='brown')

class OSMReader:

    def __init__(self, path, utmx, utmy, dim=[1000, 500]):
        self.dim = dim
        with open(path, 'r') as f:
            data = f.read()
        data = bs4.BeautifulSoup(data, "xml")
        self.nodes = data.find_all("node")
        # xmax, xmin = -float('inf'), float('inf')
        # ymax, ymin = -float('inf'), float('inf')
        for i in range(len(self.nodes)):
            self.nodes[i] = Node(self.nodes[i], utmx, utmy)
        #     xmax = max(xmax, self.nodes[i].x)
        #     xmin = min(xmin, self.nodes[i].x)
        #     ymax = max(ymax, self.nodes[i].y)
        #     ymin = min(ymin, self.nodes[i].y)
        # for i in range(len(self.nodes)):
        #     self.nodes[i].x = \
        #         (self.nodes[i].x-xmin) / (xmax-xmin) * self.dim[0] - self.dim[0]/2
        #     self.nodes[i].y = \
        #         (self.nodes[i].y-ymin) / (ymax-ymin) * self.dim[1] - self.dim[1]/2
        self.ways = data.find_all("way")
        for i in range(len(self.ways)):
            self.ways[i] = Way(self.ways[i], self.nodes)
        self.relations = data.find_all("relation")
        for i in range(len(self.relations)):
            self.relations[i] = Relation(self.relations[i], self.ways)

    def get_relation(self, id):
        for relation in self.relations:
            if relation.id == id:
                return relation
        return None

    def plot(self, show_with_colors = {}, show_all=True, ax=None):
        for relation in self.relations:
            if not show_all and relation.id not in show_with_colors.keys():
                continue
            if show_with_colors != {}:
                if relation.id in show_with_colors.keys():
                    relation.plot(show_with_colors[relation.id], ax=ax)
                else:
                    relation.plot('white', 1.0, ax=ax)
            else:
                relation.plot(ax=ax)
        if show_all:
            for way in self.ways:
                way.plot(ax=ax)
            for node in self.nodes:
                node.plot(ax=ax)

# See the lanelets
if __name__ == "__main__":
    root_path = get_package_root_path()
    josmfile = glob.glob(os.path.join(root_path, 
        "assets/exiD/*.osm"))[0]
    recordingMeta_file = glob.glob(os.path.join(root_path, 
        "assets/exiD/*_recordingMeta.csv"))[0]
    recordingMeta = pandas.read_csv(recordingMeta_file)
    utmx, utmy = float(recordingMeta["xUtmOrigin"]), float(recordingMeta["yUtmOrigin"])
    osm = OSMReader(josmfile, utmx, utmy)
    osm.plot()
    plt.show()