from tools.base import Environment
from tools.graphics import Plot2D
import copy
import glob
import os
import numpy as np
import random
from tools.utils import get_package_root_path
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tools.exid import ExiDSampleReader, frame_by_frame
import pandas
from tools.exid import OSMReader
from tools.exid import load_trajectories, load_trajectories2, lane_change_start_end, \
    find_closest_node, find_segment
from shapely.geometry import Point, LineString
from gym import spaces


class ExiDSampleEnvironmentLateral(Environment):

    def __init__(self):
        self.dt = 1/25. # from meta file
        self.ers = []
        DATA_DIR = os.path.expanduser(
            "~/Projects/Datasets/exiD/exiD-dataset-v2.0/data/")
        MAPS_DIR = os.path.expanduser(
            "~/Projects/Datasets/exiD/exiD-dataset-v2.0/maps/lanelet2/")
        nums = [item.split("/")[-1].split("_")[0] \
            for item in glob.glob(DATA_DIR+"*_tracks.csv")]
        for num in nums[:5]:
            t = DATA_DIR + num + "_tracks.csv"
            tm = DATA_DIR + num + "_tracksMeta.csv"
            rm = DATA_DIR + num + "_recordingMeta.csv"
            rmf = pandas.read_csv(rm)
            mapid = "%d" % int(rmf["locationId"])
            osmf = glob.glob(MAPS_DIR + mapid + "*.osm")[0]
            utmx, utmy = float(rmf["xUtmOrigin"]), float(rmf["yUtmOrigin"])
            osm = OSMReader(osmf, utmx, utmy)
            er = ExiDSampleReader(files = {"t": t, "tm": tm, "rm": rm})
            er.read_data(lonlat=True)
            lane_change_data = load_trajectories2(er, osm, "exidtraj/%s.pt" % num, \
                cond1 = lambda x, y: True,
                cond2 = lambda x, y: True)
            self.ers += [(er, osm, lane_change_data)]
        self.discrete = False
        high = float('inf')
        self.action_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.allowed_ers = {}
    
    @property
    def state(self):
        """
        Return the current state.
        """
        return self.curr_state

    def reset(self):
        erid = np.random.randint(len(self.ers))
        self.er, self.osm, self.lcd = self.ers[erid]
        if erid not in self.allowed_ers.keys():
            self.allowed_ers[erid] = list(range(len(self.lcd)))
        erid2 = np.random.choice(self.allowed_ers[erid])
        tid, self.startlaneletid, self.endlaneletid, startframe, endframe = \
            self.lcd[erid2]
        self.startlaneletcenters = self.osm.get_relation(self.startlaneletid).mid
        self.endlaneletcenters = self.osm.get_relation(self.endlaneletid).mid
        self.frame_dict = frame_by_frame(self.er)
        centerpts, headings, frames, vx, vy, ax, ay, _, _, llp, llco = self.er.get_track(tid, llp=True)
        self.data = {
            "tid": tid,
            "centerpts": centerpts,
            "headings": headings,
            "frames": frames,
            "vx": vx,
            "vy": vy,
            "ax": ax,
            "ay": ay,
            "llp": llp,
            "llco": llco,
        }
        self.data["startframe"], self.data["endframe"] = startframe, endframe
        closestnode, ipline = find_closest_node(self.startlaneletcenters, self.data["centerpts"][startframe][0], self.data["centerpts"][startframe][1])
        self.data["closestnode"], self.data["ipline"] = closestnode, ipline
        dr = (self.data["centerpts"][startframe][0]-closestnode.x, self.data["centerpts"][startframe][1]-closestnode.y)
        self.data["dr"] = dr
        self.data["lon_dist"], self.data["lat_dist"] = 0, 0
        self.bbox = None
        for vehicle, fid in self.frame_dict[self.data["frames"][self.data["startframe"]]]:
            if vehicle == tid:
                self.bbox = self.er.bboxes[vehicle][fid]
                break
        self.i = 0
        self.terminated = False
        self.max_i = endframe-startframe-1
        new_point = self.data["ipline"].interpolate(self.data["lon_dist"])
        segpt1, segpt2 = find_segment(self.data["ipline"], new_point)
        lonveclength = ((segpt1[0]-segpt2[0])**2+(segpt1[1]-segpt2[1])**2)**0.5
        dx = (segpt2[0]-segpt1[0])/lonveclength
        dy = (segpt2[1]-segpt1[1])/lonveclength
        dx *= -1; dy *= 1 # rotate unit
        newx = new_point.x+dy*self.data["lat_dist"]; newx += self.data["dr"][0]
        newy = new_point.y+dx*self.data["lat_dist"]; newy += self.data["dr"][1]
        targetpts = []
        for node in self.endlaneletcenters:
            targetpts += [(node.x, node.y)]
        ipline2 = LineString(targetpts)
        rightside = ipline2.buffer(100, single_sided=True)
        side = -1 if rightside.contains(Point(newx, newy)) else 1
        new_point2 = ipline2.interpolate(self.data["lon_dist"])
        dist_target = Point(newx, newy).distance(new_point2)
        self.curr_state = np.array([side*dist_target])
        if dist_target > 6:
            self.allowed_ers[erid].remove(erid2)
            # print(["%s:%d" % (k, len(v)) for k, v in self.allowed_ers.items()])
            return self.reset()
        else:
            return self.curr_state

    def seed(self, s=None):
        """
        Seed this environment.
        """
        random.seed(s)
        np.random.seed(s)

    def step(self, action=None):
        self.i += 1
        self.terminated = self.i > self.max_i
        new_point = self.data["ipline"].interpolate(self.data["lon_dist"])
        segpt1, segpt2 = find_segment(self.data["ipline"], new_point)
        lonveclength = ((segpt1[0]-segpt2[0])**2+(segpt1[1]-segpt2[1])**2)**0.5
        dx = (segpt2[0]-segpt1[0])/lonveclength
        dy = (segpt2[1]-segpt1[1])/lonveclength
        dc = (self.data["centerpts"][self.data["startframe"]+self.i, 0]-self.data["centerpts"][self.data["startframe"]+self.i-1, 0], self.data["centerpts"][self.data["startframe"]+self.i, 1]-self.data["centerpts"][self.data["startframe"]+self.i-1, 1])
        dclon = dc[0]*dx+dc[1]*dy
        self.data["lon_dist"] += dclon
        dx *= -1; dy *= 1 # rotate unit
        dclat = dc[0]*dy+dc[1]*dx
        info = {}
        if action is not None:
            action = float(action)
            dclat = action*self.dt
        else:
            info = {"action": dclat/self.dt}
        self.data["lat_dist"] += dclat
        newx = new_point.x+dy*self.data["lat_dist"]; newx += self.data["dr"][0]
        newy = new_point.y+dx*self.data["lat_dist"]; newy += self.data["dr"][1]
        targetpts = []
        for node in self.endlaneletcenters:
            targetpts += [(node.x, node.y)]
        ipline2 = LineString(targetpts)
        rightside = ipline2.buffer(100, single_sided=True)
        side = -1 if rightside.contains(Point(newx, newy)) else 1
        new_point2 = ipline2.interpolate(self.data["lon_dist"])
        dist_target = Point(newx, newy).distance(new_point2)
        self.curr_state = np.array([side*dist_target])
        return {
            "next_state": self.curr_state,
            "reward": 1-dist_target/10. if dist_target <= 10. else 0.,
            "done": self.terminated, 
            "info": info,
        }

    def render(self, **kwargs):
        new_point = self.data["ipline"].interpolate(self.data["lon_dist"])
        segpt1, segpt2 = find_segment(self.data["ipline"], new_point)
        lonveclength = ((segpt1[0]-segpt2[0])**2+(segpt1[1]-segpt2[1])**2)**0.5
        dx = (segpt2[0]-segpt1[0])/lonveclength
        dy = (segpt2[1]-segpt1[1])/lonveclength
        dx *= -1; dy *= 1 # rotate unit
        newx = new_point.x+dy*self.data["lat_dist"]; newx += self.data["dr"][0]
        newy = new_point.y+dx*self.data["lat_dist"]; newy += self.data["dr"][1]
        dp = (newx-self.data["centerpts"][self.data["startframe"]][0], \
            newy-self.data["centerpts"][self.data["startframe"]][1])
        new_bbox = copy.deepcopy(self.bbox)
        new_bbox[:, 0] += dp[0]
        new_bbox[:, 1] += dp[1]
        self.display_bbox = new_bbox
        if not hasattr(self, "plot"):
            self.plot = Plot2D({
                "env": lambda p, l, t: self,
            }, [
                [
                    lambda p, l, t: not l["env"].terminated,
                    lambda p, l, o, t: p.polygon(l["env"].display_bbox, o=o, facecolor="brown")
                ],
            ], mode="dynamic", interval=50)
            self.plot.ax.axis("equal")
            self.osm.plot({self.startlaneletid:'cyan', self.endlaneletid:'green'}, 
                show_all=False, ax=self.plot.ax)
        self.plot.show(block=False)
        if "mode" in kwargs.keys() and kwargs["mode"] == "rgb_array":
            self.plot.fig.canvas.draw()
            img = np.frombuffer(self.plot.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.plot.fig.canvas.get_width_height()[::-1] + (3,))
            return img
