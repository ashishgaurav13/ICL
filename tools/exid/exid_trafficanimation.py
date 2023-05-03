import glob
import pandas as pd
import os
import numpy as np
from tools.utils import get_package_root_path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation

class ExiDSampleReader:
    """
    Reads the ExiD sample data. Provides utilities around it.
    """

    def __init__(self, files={}, dim=[1000, 500], dt=0.1):
        """
        Initialize the ExiDSampleReader.
        """
        self.F = 0
        self.dim = dim
        root_path = get_package_root_path()
        tracks_files = glob.glob(os.path.join(root_path, 
            "assets/exiD/*_tracks.csv"))
        tracksMeta_files = glob.glob(os.path.join(root_path, 
            "assets/exiD/*_tracksMeta.csv"))
        recordingMeta_files = glob.glob(os.path.join(root_path, 
            "assets/exiD/*_recordingMeta.csv"))
        self.dt = dt
        if "t" in files.keys() and "tm" in files.keys() and "rm" in files.keys():
            self.tracks_file, self.tracksMeta_file, self.recordingMeta_file = \
                files["t"], files["tm"], files["rm"]
        else:
            self.tracks_file, self.tracksMeta_file, self.recordingMeta_file = \
                tracks_files[self.F], tracksMeta_files[self.F], \
                recordingMeta_files[self.F]

    def calculate_rotated_bboxes(self, center_points_x, center_points_y, length,
        width, rotation=0):
        """
        Calculate bounding box vertices from centroid, width and length.
        """
        centroid = np.array([center_points_x, center_points_y]).transpose()
        centroid = np.array(centroid)
        if centroid.shape == (2,):
            centroid = np.array([centroid])
        data_length = centroid.shape[0]
        rotated_bbox_vertices = np.empty((data_length, 4, 2))
        rotated_bbox_vertices[:, 0, 0] = -length / 2
        rotated_bbox_vertices[:, 0, 1] = -width / 2
        rotated_bbox_vertices[:, 1, 0] = length / 2
        rotated_bbox_vertices[:, 1, 1] = -width / 2
        rotated_bbox_vertices[:, 2, 0] = length / 2
        rotated_bbox_vertices[:, 2, 1] = width / 2
        rotated_bbox_vertices[:, 3, 0] = -length / 2
        rotated_bbox_vertices[:, 3, 1] = width / 2
        for i in range(4):
            th, r = self.cart2pol(rotated_bbox_vertices[:, i, :])
            rotated_bbox_vertices[:, i, :] = self.pol2cart(th + rotation, r).squeeze()
            rotated_bbox_vertices[:, i, :] = rotated_bbox_vertices[:, i, :] + centroid
        return rotated_bbox_vertices

    def cart2pol(self, cart):
        """
        Transform cartesian to polar coordinates.
        """
        if cart.shape == (2,):
            cart = np.array([cart])
        x = cart[:, 0]
        y = cart[:, 1]
        th = np.arctan2(y, x)
        r = np.sqrt(np.power(x, 2) + np.power(y, 2))
        return th, r

    def pol2cart(self, th, r):
        """
        Transform polar to cartesian coordinates.
        """
        x = np.multiply(r, np.cos(th))
        y = np.multiply(r, np.sin(th))
        cart = np.array([x, y]).transpose()
        return cart

    def read_data(self, lonlat=False):
        """
        Read the data, normalize it and store it.
        """
        self.tm = pd.read_csv(self.tracksMeta_file).to_dict(orient="records")
        self.rm = pd.read_csv(self.recordingMeta_file).to_dict(orient="records")
        self.t = pd.read_csv(self.tracks_file).groupby(["trackId"], sort=False)
        # xmin, xmax = np.inf, -np.inf
        # ymin, ymax = np.inf, -np.inf
        self.bboxes = []
        self.centerpts = []
        self.frames = []
        self.headings = []
        self.speedxs, self.speedys = [], []
        self.accxs, self.accys = [], []
        self.llp = []
        self.llco = []
        for gid, rows in self.t:
            g = rows
            xCenterVis = np.array(g["xCenter"].tolist())
            yCenterVis = np.array(g["yCenter"].tolist())
            centerVis = np.stack([np.array(g["xCenter"].tolist()), 
                np.array(g["yCenter"].tolist())], axis=-1)
            widthVis = np.array(g["width"].tolist())
            heightVis = np.array(g["length"].tolist())
            headingVis = np.array(g["heading"].tolist())
            bboxVis = self.calculate_rotated_bboxes(
                xCenterVis, yCenterVis,
                heightVis, widthVis, np.deg2rad(headingVis)
            )
            # bbox = bboxVis/(0.10106*4)
            # centerpt = centerVis/(0.10106*4)
            bbox = bboxVis
            centerpt = centerVis
            self.bboxes += [bbox]
            self.centerpts += [centerpt]
            self.headings += [headingVis]
            self.frames += [np.array(g["frame"].tolist())]
            if lonlat:
                gg = g["lonLaneletPos"].tolist()
                ff = g["latLaneCenterOffset"].tolist()
                for ggi in range(len(gg)):
                    gg[ggi] = float(gg[ggi].split(";")[0])
                    ff[ggi] = float(ff[ggi].split(";")[0])
                self.llp += [np.array(gg)]
                self.llco += [np.array(ff)]
                self.speedxs += [np.array(g["lonVelocity"].tolist())]
                self.speedys += [np.array(g["latVelocity"].tolist())]
                self.accxs += [np.array(g["lonAcceleration"].tolist())]
                self.accys += [np.array(g["latAcceleration"].tolist())]
            else:
                self.speedxs += [np.array(g["xVelocity"].tolist())]
                self.speedys += [np.array(g["yVelocity"].tolist())]
                self.accxs += [np.array(g["xAcceleration"].tolist())]
                self.accys += [np.array(g["yAcceleration"].tolist())]

    def get_track(self, track_num, llp=False):
        """
        Return centerpts and angle of car for a certain track_num.
        """
        # combinations = [[0, 1], [1, 2], [2, 3], [3, 0]]
        # euclidean = lambda x1, y1, x2, y2: np.sqrt((x1-x2)**2+(y1-y2)**2)
        # min_h, max_w = np.inf, -np.inf
        # for bbox in self.bboxes[track_num]:
        #     lengths = []
        #     for i, j in combinations:
        #         lengths += [euclidean(bbox[i, 0], bbox[i, 1], bbox[j, 0], bbox[j, 1])]
        #     min_h = min(min_h, np.min(lengths))
        #     max_w = max(max_w, np.max(lengths))
        extra = []
        if llp:
            extra += [self.llp[track_num], self.llco[track_num]]
        return self.centerpts[track_num], self.headings[track_num], \
            self.frames[track_num], \
            self.speedxs[track_num], self.speedys[track_num],\
            self.accxs[track_num], self.accys[track_num],\
            None, None, *extra 

# Where do these tracks start?
def show_all_starts(er):
    startxs, startys = [], []
    for i in range(len(er.bboxes)):
        cpts, _, _, _, _, _, _, _, _ = er.get_track(i)
        startx, starty = cpts[0, :]
        startxs += [startx]
        startys += [starty]
    plt.scatter(startxs, startys)
    plt.show()
# show_all_starts()


# Given some start and/or end condition, show the start and end points of
# tracks that satisfy these
def show_start_end(er, accept_filter_start, accept_filter_end = None):
    startxs, startys = [], []
    endxs, endys = [], []
    for i in range(len(er.bboxes)):
        cpts, _, _, _, _, _, _, _, _ = er.get_track(i)
        startx, starty = cpts[0, :]
        if not accept_filter_start(startx, starty):
            continue
        endx, endy = cpts[-1, :]
        if accept_filter_end and not accept_filter_end(endx, endy):
            continue
        startxs += [startx]
        startys += [starty]
        endxs += [endx]
        endys += [endy]
    print(len(startxs))
    plt.scatter(startxs, startys, color='blue')
    plt.scatter(endxs, endys, color='red')
    plt.show()
# show_start_end(lambda x, y: y < -250, lambda x, y: x > 550)


# Instead of showing, return the track ids
def get_trackids(er, accept_filter_start, accept_filter_end = None):
    trackids = set([])
    for i in range(len(er.bboxes)):
        cpts, _, _, _, _, _, _, _, _ = er.get_track(i)
        startx, starty = cpts[0, :]
        if not accept_filter_start(startx, starty):
            continue
        endx, endy = cpts[-1, :]
        if accept_filter_end and not accept_filter_end(endx, endy):
            continue
        trackids.add(i)
    return trackids
# suitable_trackids = get_trackids(lambda x, y: y < -250, lambda x, y: x > 550)
# print(suitable_trackids)


# Get a dict in which the keys are the frame ids and values are a list of vehicle
# ids present in that frame
def frame_by_frame(er):
    frame_dict = {}
    for i in range(len(er.bboxes)):
        _, _, frames, _, _, _, _, _, _ = er.get_track(i)
        for fid, frame in enumerate(frames):
            if frame not in frame_dict:
                frame_dict[frame] = set([])
            frame_dict[frame].add((i, fid))
    return frame_dict
# frame_dict = frame_by_frame()
# print(len(frame_dict))


if __name__ == "__main__":

    # Read the track data
    er = ExiDSampleReader()
    er.read_data()
    # How many tracks? One track per vehicle
    print(len(er.bboxes))

    # Animate a particular track
    import pandas
    from exid_lanelet import OSMReader
    root_path = get_package_root_path()
    josmfile = glob.glob(os.path.join(root_path, 
        "assets/exiD/*.osm"))[0]
    recordingMeta_file = glob.glob(os.path.join(root_path, 
        "assets/exiD/*_recordingMeta.csv"))[0]
    recordingMeta = pandas.read_csv(recordingMeta_file)
    utmx, utmy = float(recordingMeta["xUtmOrigin"]), float(recordingMeta["yUtmOrigin"])
    osm = OSMReader(josmfile, utmx, utmy)
    osm.plot()
    # Next, show the vehicles for a particular track
    suitable_trackids = get_trackids(er, lambda x, y: y < -250, lambda x, y: x > 550)
    tid = list(suitable_trackids)[0]
    frame_dict = frame_by_frame(er)
    _, _, frames, _, _, _, _, _, _ = er.get_track(tid)
    vehicleobjs = {}
    def animate(i):
        present_vehicles = []
        blit_objs = []
        for vehicle, fid in frame_dict[frames[i]]:
            bbox = er.bboxes[vehicle][fid]
            present_vehicles += [vehicle]
            if vehicle == tid:
                fc = 'brown'
            else:
                fc = 'black'
            if vehicle in vehicleobjs.keys():
                vehicleobjs[vehicle].set_xy(bbox)
            else:
                vehicleobjs[vehicle] = Polygon(bbox, facecolor=fc)
                plt.gca().add_patch(vehicleobjs[vehicle])
            blit_objs += [vehicleobjs[vehicle]]
        extra_vehicles = set(vehicleobjs.keys())-set(present_vehicles)
        for vehicle in extra_vehicles:
            del vehicleobjs[vehicle]
        return blit_objs
    ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate,
        frames=len(frames), interval=1, blit=True, repeat=False)
    # ani.save("sampletracks.gif", writer=matplotlib.animation.ImageMagickWriter(fps=25,
    #             extra_args=['-loop', '0']),
    #             progress_callback=lambda i, n: print("%d/%d" % (i, n)))
    plt.show()