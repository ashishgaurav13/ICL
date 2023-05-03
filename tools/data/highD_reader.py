import glob
import pandas as pd
import os
import numpy as np
from tools.utils import get_package_root_path

class HighDSampleReader:
    """
    Reads the HighD sample data. Provides utilities around it.
    """

    def __init__(self, F=0, dim=[1000, 100], dt=0.1):
        """
        Initialize the HighDSampleReader.
        """
        self.F = 0
        self.dim = dim
        root_path = get_package_root_path()
        tracks_files = glob.glob(os.path.join(root_path, 
            "assets/highD/*_tracks.csv"))
        tracksMeta_files = glob.glob(os.path.join(root_path, 
            "assets/highD/*_tracksMeta.csv"))
        recordingMeta_files = glob.glob(os.path.join(root_path, 
            "assets/highD/*_recordingMeta.csv"))
        self.dt = dt
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

    def read_data(self):
        """
        Read the data, normalize it and store it.
        """
        self.tm = pd.read_csv(self.tracksMeta_file).to_dict(orient="records")
        self.rm = pd.read_csv(self.recordingMeta_file).to_dict(orient="records")
        self.t = pd.read_csv(self.tracks_file).groupby(["id"], sort=False)
        xmin, xmax = np.inf, -np.inf
        ymin, ymax = np.inf, -np.inf
        self.bboxes = []
        self.centerpts = []
        self.frames = []
        self.headings = []
        self.speedxs, self.speedys = [], []
        self.accxs, self.accys = [], []
        for gid, rows in self.t:
            g = rows
            xCenterVis = np.array(g["x"].tolist())
            yCenterVis = -np.array(g["y"].tolist())
            centerVis = np.stack([np.array(g["x"].tolist()), 
                -np.array(g["y"].tolist())], axis=-1)
            widthVis = np.array(g["width"].tolist())
            heightVis = np.array(g["height"].tolist())
            bboxVis = self.calculate_rotated_bboxes(
                xCenterVis, yCenterVis,
                heightVis, widthVis,
            )
            bbox = bboxVis/(0.10106*4)
            centerpt = centerVis/(0.10106*4)
            self.bboxes += [bbox]
            self.centerpts += [centerpt]
            self.headings += [np.zeros_like(xCenterVis)]
            self.frames += [np.array(g["frame"].tolist())]
            self.speedxs += [np.array(g["xVelocity"].tolist())]
            self.speedys += [np.array(g["yVelocity"].tolist())]
            self.accxs += [np.array(g["xAcceleration"].tolist())]
            self.accys += [np.array(g["yAcceleration"].tolist())]
            xmin, xmax = min(xmin, np.min(bbox[:, :, 0])), \
                max(xmax, np.max(bbox[:, :, 0]))
            ymin, ymax = min(ymin, np.min(bbox[:, :, 1])), \
                max(ymax, np.max(bbox[:, :, 1]))
        for i in range(len(self.bboxes)):
            self.bboxes[i][:, :, 0] = \
                (self.bboxes[i][:, :, 0]-xmin) / (xmax-xmin) * self.dim[0] - self.dim[0]/2
            self.bboxes[i][:, :, 1] = \
                (self.bboxes[i][:, :, 1]-ymin) / (ymax-ymin) * self.dim[1] - self.dim[1]/2
            self.centerpts[i][:, 0] = \
                (self.centerpts[i][:, 0]-xmin) / (xmax-xmin) * self.dim[0] - self.dim[0]/2
            self.centerpts[i][:, 1] = \
                (self.centerpts[i][:, 1]-ymin) / (ymax-ymin) * self.dim[1] - self.dim[1]/2
            self.speedxs[i] = \
                (self.speedxs[i]) / (xmax-xmin) * self.dim[0]
            self.speedys[i] = \
                (self.speedys[i]) / (ymax-ymin) * self.dim[1]
            self.accxs[i] = \
                (self.accxs[i]) / (xmax-xmin) * self.dim[0]
            self.accys[i] = \
                (self.accys[i]) / (ymax-ymin) * self.dim[1]

    def get_track(self, track_num):
        """
        Return centerpts and angle of car for a certain track_num.
        """
        combinations = [[0, 1], [1, 2], [2, 3], [3, 0]]
        euclidean = lambda x1, y1, x2, y2: np.sqrt((x1-x2)**2+(y1-y2)**2)
        min_h, max_w = np.inf, -np.inf
        for bbox in self.bboxes[track_num]:
            lengths = []
            for i, j in combinations:
                lengths += [euclidean(bbox[i, 0], bbox[i, 1], bbox[j, 0], bbox[j, 1])]
            min_h = min(min_h, np.min(lengths))
            max_w = max(max_w, np.max(lengths))
        return self.centerpts[track_num], self.headings[track_num], \
            self.frames[track_num], \
            self.speedxs[track_num], self.speedys[track_num],\
            self.accxs[track_num], self.accys[track_num],\
            min_h, max_w 
    
    def get_best_start(self, gap=1000, frames_len_max=600, condition=None):
        """
        Considering all trajectories of length frames_len_max or less, find the 
        start frame number such that we have the max number of trajectories
        within a gap interval (all these trajectories end within the interval).
        """
        min_maxs = []
        for i in range(len(self.bboxes)):
            fr = self.frames[i]
            cp = self.centerpts[i]
            ang = self.headings[i]
            if len(fr) > frames_len_max: continue
            if np.isnan(fr).any() or np.isnan(cp).any() or np.isnan(ang).any(): continue
            min_maxs += [(np.min(fr), np.max(fr), i)]
            frames = np.array(fr)
        min_maxs = sorted(min_maxs, key=lambda x: x[0])
        ret = []
        for i in range(len(min_maxs)):
            count = 0
            for j in range(i, len(min_maxs)):
                if 0 <= min_maxs[j][1] - min_maxs[i][0] <= gap:
                    count += 1
            ret += [(min_maxs[i][0], count, min_maxs[i][-1])]
        ret = sorted(ret, key=lambda x: x[1])[::-1]
        for item in ret:
            if condition != None and condition(self.centerpts[item[-1]],\
                self.frames[item[-1]]):
                return item[0]
            elif condition == None:
                return item[0]
        return None