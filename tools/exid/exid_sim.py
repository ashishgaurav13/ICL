import copy
import glob
import os
import numpy as np
from tools.utils import get_package_root_path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation
import matplotlib.pyplot as plt
from .exid_trafficanimation import ExiDSampleReader, get_trackids, frame_by_frame
import pandas
from .exid_lanelet import OSMReader
import tqdm
from .exid_localize import localize, is_adjacent_lane
import torch
from shapely.geometry import LineString

startlaneletid = '1412'
endlaneletid = '1411'
dt = 1/25. # from meta file
trajectories_pickle_file = 'traj.pt' # will create if does not exist

def load_trajectories(er, osm, trajectories_pickle_file, cond1 = lambda x, y: y < -250,
    cond2 = lambda x, y: x > 550):
    if os.path.exists(trajectories_pickle_file):
        return torch.load(trajectories_pickle_file)
    suitable_trackids = get_trackids(er, cond1, cond2)
    tracks_with_one_lane_change = []
    lanelet_ids_lane_change = {}
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
    torch.save(tracks_with_one_lane_change, trajectories_pickle_file)
    return tracks_with_one_lane_change

def load_trajectories2(er, osm, trajectories_pickle_file, cond1 = lambda x, y: y < -250,
    cond2 = lambda x, y: x > 550):
    if os.path.exists(trajectories_pickle_file):
        return torch.load(trajectories_pickle_file)
    suitable_trackids = get_trackids(er, cond1, cond2)
    tracks_with_one_lane_change = []
    lanelet_ids_lane_change = {}
    tracks_lanelets = {}
    for tid in tqdm.tqdm(list(suitable_trackids)):
        localizations = []
        centerpts, _, _, _, _, _, _, _, _ = er.get_track(tid)
        lane_changes = 0
        lanechangesset = set([])
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
                        lanechangesset.add((localizations[-1].id, where.id))
                    localizations += [where]
        if lane_changes == 1:
            tracks_with_one_lane_change += [tid]
            tracks_lanelets[tid] = lanechangesset
    all_data = []
    for tid in tracks_lanelets.keys():
        centerpts, headings, frames, vx, vy, ax, ay, _, _, llp, llco = er.get_track(tid, llp=True)
        for (sid, eid) in tracks_lanelets[tid]:
            sf, ef = lane_change_start_end2(osm, centerpts, sid, eid)
            if sf != None and ef != None:
                all_data += [(tid, sid, eid, sf, ef)]
    torch.save(all_data, trajectories_pickle_file)
    return all_data

def lane_change_start_end(osm, centerpts):
    startframe, endframe = None, None
    targetlanestarted = False
    for i in range(centerpts.shape[0]):
        where = localize(osm, centerpts[i][0], centerpts[i][1])
        if where.id == startlaneletid and startframe is None:
            startframe = i
        if where.id == endlaneletid and not targetlanestarted:
            targetlanestarted = True
        if where.id != endlaneletid and targetlanestarted:
            endframe = i-1
            break
    return startframe, endframe

def lane_change_start_end2(osm, centerpts, startlaneletid, endlaneletid):
    startframe, endframe = None, None
    targetlanestarted = False
    for i in range(centerpts.shape[0]):
        where = localize(osm, centerpts[i][0], centerpts[i][1])
        if where.id == startlaneletid and startframe is None:
            startframe = i
        if where.id == endlaneletid and not targetlanestarted:
            targetlanestarted = True
        if where.id != endlaneletid and targetlanestarted:
            endframe = i-1
            break
    return startframe, endframe

# find closest centerline point to ego's start position, then dr = ego's pos - this point
# return centerpoint linestring from closest node
def find_closest_node(laneletcenters, x, y, onlynode=False):
    closestnode = None
    mindist = float('inf')
    for node in laneletcenters:
        newdist = ((node.x-x)**2+(node.y-y)**2)**0.5
        if newdist < mindist:
            mindist = newdist
            closestnode = node
    if onlynode:
        return closestnode
    points = []
    has_started = False
    for node in laneletcenters:
        if node == closestnode and not has_started:
            has_started = True
        if has_started:
            points += [(node.x, node.y)]
    return closestnode, LineString(points)

def find_segment(ipline, new_point):
    all_dist = []
    for coord_i, coord in enumerate(list(ipline.coords)):
        all_dist += [(coord, ((coord[0]-new_point.x)**2+(coord[1]-new_point.y)**2)**0.5, coord_i)]
    all_dist = sorted(all_dist, key=lambda item: item[1])
    if all_dist[0][-1] < all_dist[1][-1]:
        return [all_dist[0][0], all_dist[1][0]]
    else:
        return [all_dist[1][0], all_dist[0][0]]

def find_segment2(ipline, new_pointx, new_pointy):
    all_dist = []
    for coord_i, coord in enumerate(ipline):
        all_dist += [(coord, ((coord.x-new_pointx)**2+(coord.y-new_pointy)**2)**0.5, coord_i)]
    all_dist = sorted(all_dist, key=lambda item: item[1])
    if all_dist[0][-1] < all_dist[1][-1]:
        return [all_dist[0][0], all_dist[1][0]]
    else:
        return [all_dist[1][0], all_dist[0][0]]

if __name__ == "__main__":
    er = ExiDSampleReader()
    er.read_data(lonlat=True)
    root_path = get_package_root_path()
    josmfile = glob.glob(os.path.join(root_path, 
        "assets/exiD/*.osm"))[0]
    recordingMeta_file = glob.glob(os.path.join(root_path, 
        "assets/exiD/*_recordingMeta.csv"))[0]
    recordingMeta = pandas.read_csv(recordingMeta_file)
    utmx, utmy = float(recordingMeta["xUtmOrigin"]), float(recordingMeta["yUtmOrigin"])
    osm = OSMReader(josmfile, utmx, utmy)
    startlaneletcenters = osm.get_relation(startlaneletid).mid
    endlaneletcenters = osm.get_relation(endlaneletid).mid
    frame_dict = frame_by_frame(er)
    osm.plot({startlaneletid:'cyan', endlaneletid:'green'}, show_all=False)
    tid = load_trajectories(er, osm, trajectories_pickle_file)[2]
    centerpts, headings, frames, vx, vy, ax, ay, _, _, llp, llco = er.get_track(tid, llp=True)
    startframe, endframe = lane_change_start_end(osm, centerpts)
    theta = np.deg2rad(headings[0])
    closestnode, ipline = find_closest_node(startlaneletcenters, centerpts[startframe][0], centerpts[startframe][1])
    dr = (centerpts[startframe][0]-closestnode.x, centerpts[startframe][1]-closestnode.y)
    is_driving = True
    driving_actions = [0.,]*(endframe-startframe+1) # lateral velocity
    lon_dist, lat_dist = 0, 0
    latv = None
    vehicleobj = None
    bbox = None
    def animate(i):
        global vehicleobj, bbox, closestnode, ipline
        global lon_dist, lat_dist
        global latv
        blit_objs = []
        if (not is_driving) or (is_driving and i == 0):
            for vehicle, fid in frame_dict[frames[startframe+i]]:
                if vehicle == tid:
                    bbox = er.bboxes[vehicle][fid]
                    latv = vy[startframe+i]
                    break
        # print("i=",i)
        if is_driving and i > 0:
            new_point = ipline.interpolate(lon_dist)
            segpt1, segpt2 = find_segment(ipline, new_point)
            lonveclength = ((segpt1[0]-segpt2[0])**2+(segpt1[1]-segpt2[1])**2)**0.5
            dx = (segpt2[0]-segpt1[0])/lonveclength
            dy = (segpt2[1]-segpt1[1])/lonveclength
            dc = (centerpts[startframe+i, 0]-centerpts[startframe+i-1, 0], centerpts[startframe+i, 1]-centerpts[startframe+i-1, 1])
            dclon = dc[0]*dx+dc[1]*dy
            lon_dist += dclon
            dx *= -1; dy *= 1 # rotate unit
            dclat = dc[0]*dy+dc[1]*dx
            lat_dist += dclat
            newx = new_point.x+dy*lat_dist; newx += dr[0]
            newy = new_point.y+dx*lat_dist; newy += dr[1]
            dp = (newx-centerpts[startframe][0], newy-centerpts[startframe][1])
            new_bbox = copy.deepcopy(bbox)
            new_bbox[:, 0] += dp[0]
            new_bbox[:, 1] += dp[1]
        fc = 'brown'
        if vehicleobj is not None:
            if is_driving and i > 0:
                vehicleobj.set_xy(new_bbox)
            else:
                vehicleobj.set_xy(bbox)
        else:
            if is_driving and i > 0:
                vehicleobj = Polygon(new_bbox, facecolor=fc)
            else:
                vehicleobj = Polygon(bbox, facecolor=fc)
            plt.gca().add_patch(vehicleobj)
        blit_objs = [vehicleobj]
        return blit_objs
    plt.axis('equal')
    ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate,
        frames=endframe-startframe, interval=1, blit=True, repeat=False) # 1 less length than usual
    ani.save("randomsampletrack3.gif", writer=matplotlib.animation.ImageMagickWriter(fps=25,
                extra_args=['-loop', '0']),
                progress_callback=lambda i, n: print("%d/%d" % (i, n)))
    # plt.show()