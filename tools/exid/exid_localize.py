import glob
import os
from threading import local
import numpy as np
from tools.utils import get_package_root_path
from .exid_trafficanimation import ExiDSampleReader, get_trackids, frame_by_frame
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt

# pip install shapely tqdm

def localize(osm, x, y):
    point = Point(x, y)
    for relation in osm.relations:
        polygon = Polygon(list(map(lambda node: (node.x, node.y),
            relation.data[0].data+relation.data[1].data[::-1])))
        if polygon.contains(point):
            return relation
    return None

def is_adjacent_lane(osm, relation1, relation2):
    points1 = set(filter(lambda node: (node.x, node.y),
            relation1.data[0].data+relation1.data[1].data[::-1]))
    points2 = set(filter(lambda node: (node.x, node.y),
            relation2.data[0].data+relation2.data[1].data[::-1]))
    startlaneletcenters = relation1.mid
    endlaneletcenters = relation2.mid
    mindist = float('inf')
    for node1 in startlaneletcenters:
        for node2 in startlaneletcenters:
            dist = (node1.x-node2.x)**2+(node1.y-node2.y)**2
            dist = dist**0.5
            mindist = min(mindist, dist)
    num_shared_points = len(points1.intersection(points2))
    if num_shared_points <= 2 or mindist > 10:
        return False
    else:
        return True

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
    # osm.plot()
    # Next, show the vehicles for a particular track
    suitable_trackids = get_trackids(er, lambda x, y: y < -250, lambda x, y: x > 550)
    # Check lane changes
    frame_dict = frame_by_frame(er)
    tracks_with_one_lane_change = []
    lanelet_ids_lane_change = {}
    import tqdm
    for tid in tqdm.tqdm(list(suitable_trackids)):
        localizations = []
        centerpts, _, _, _, _, _, _, _, _ = er.get_track(tid)
        lane_changes = 0
        for cid in range(centerpts.shape[0]):
            cx, cy = centerpts[cid, 0], centerpts[cid, 1]
            where = localize(osm, cx, cy)
            if len(localizations) == 0:
                localizations += [where]
            else:
                if where.id != localizations[-1].id:
                    are_adjacent = is_adjacent_lane(osm, where, localizations[-1])
                    if are_adjacent:
                        lane_changes += 1
                        if (localizations[-1].id, where.id) not in lanelet_ids_lane_change.keys():
                            lanelet_ids_lane_change[(localizations[-1].id, where.id)] = 0
                        lanelet_ids_lane_change[(localizations[-1].id, where.id)] += 1
                    localizations += [where]
        if lane_changes == 1:
            tracks_with_one_lane_change += [tid]
    # print(tracks_with_one_lane_change)
    # print(lanelet_ids_lane_change)
    # print(len(tracks_with_one_lane_change))
    # osm.plot({'1412':'cyan', '1411':'green'}); plt.show()
    osm.plot({'1412':'cyan', '1411':'green'}, show_all=False); plt.show()
