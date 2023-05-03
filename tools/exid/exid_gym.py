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
from .exid_trafficanimation import ExiDSampleReader, frame_by_frame
import pandas
from .exid_lanelet import OSMReader
from .exid_sim import load_trajectories, load_trajectories2, lane_change_start_end, \
    find_closest_node, find_segment, find_segment2
from shapely.geometry import Point, LineString

class ExiDSampleEnvironmentLateral(Environment):

    def __init__(self):
        self.startlaneletid = '1412'
        self.endlaneletid = '1411'
        self.dt = 1/25. # from meta file
        self.trajectories_pickle_file = 'traj.pt' # will create if does not exist
        self.er = ExiDSampleReader()
        self.er.read_data(lonlat=True)
        root_path = get_package_root_path()
        josmfile = glob.glob(os.path.join(root_path, 
            "assets/exiD/*.osm"))[0]
        recordingMeta_file = glob.glob(os.path.join(root_path, 
            "assets/exiD/*_recordingMeta.csv"))[0]
        recordingMeta = pandas.read_csv(recordingMeta_file)
        utmx, utmy = float(recordingMeta["xUtmOrigin"]), float(recordingMeta["yUtmOrigin"])
        self.osm = OSMReader(josmfile, utmx, utmy)
        self.startlaneletcenters = self.osm.get_relation(self.startlaneletid).mid
        self.endlaneletcenters = self.osm.get_relation(self.endlaneletid).mid
        self.frame_dict = frame_by_frame(self.er)
        self.trajs = load_trajectories(self.er, self.osm, self.trajectories_pickle_file)
    
    @property
    def state(self):
        """
        Return the current state.
        """
        return self.curr_state

    def reset(self):
        tid = self.trajs[np.random.randint(len(self.trajs))]
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
        startframe, endframe = lane_change_start_end(self.osm, self.data["centerpts"])
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
        new_point2 = ipline2.interpolate(self.data["lon_dist"])
        dist_target = Point(newx, newy).distance(new_point2)
        self.curr_state = np.array([dist_target])
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
        if action is not None:
            dclat = action*self.dt
        self.data["lat_dist"] += dclat
        newx = new_point.x+dy*self.data["lat_dist"]; newx += self.data["dr"][0]
        newy = new_point.y+dx*self.data["lat_dist"]; newy += self.data["dr"][1]
        targetpts = []
        for node in self.endlaneletcenters:
            targetpts += [(node.x, node.y)]
        ipline2 = LineString(targetpts)
        new_point2 = ipline2.interpolate(self.data["lon_dist"])
        dist_target = Point(newx, newy).distance(new_point2)
        self.curr_state = np.array([dist_target])
        return {
            "next_state": self.curr_state,
            "reward": -dist_target,
            "done": self.terminated, 
            "info": {}
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

class ExiDSampleEnvironmentLateral2(Environment):

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
            lane_change_data = load_trajectories2(er, osm, "assets/exiD/%s.pt" % num, \
                cond1 = lambda x, y: True,
                cond2 = lambda x, y: True)
            self.ers += [(er, osm, lane_change_data)]
    
    @property
    def state(self):
        """
        Return the current state.
        """
        return self.curr_state

    def reset(self):
        self.er, self.osm, self.lcd = self.ers[np.random.randint(len(self.ers))]
        tid, self.startlaneletid, self.endlaneletid, startframe, endframe = \
            self.lcd[np.random.randint(len(self.lcd))]
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
        new_point2 = ipline2.interpolate(self.data["lon_dist"])
        dist_target = Point(newx, newy).distance(new_point2)
        self.curr_state = np.array([dist_target])
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
        if action is not None:
            dclat = action*self.dt
        self.data["lat_dist"] += dclat
        newx = new_point.x+dy*self.data["lat_dist"]; newx += self.data["dr"][0]
        newy = new_point.y+dx*self.data["lat_dist"]; newy += self.data["dr"][1]
        targetpts = []
        for node in self.endlaneletcenters:
            targetpts += [(node.x, node.y)]
        ipline2 = LineString(targetpts)
        new_point2 = ipline2.interpolate(self.data["lon_dist"])
        dist_target = Point(newx, newy).distance(new_point2)
        self.curr_state = np.array([dist_target])
        return {
            "next_state": self.curr_state,
            "reward": -dist_target,
            "done": self.terminated, 
            "info": {}
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

if __name__ == "__main__":
    env = ExiDSampleEnvironmentLateral2()
    obs = env.reset()
    done = False
    while not done:
        env.render()
        next_step = env.step(None)
        obs, reward, done, info = next_step["next_state"], next_step["reward"],\
            next_step["done"], next_step["info"]
        print(reward)