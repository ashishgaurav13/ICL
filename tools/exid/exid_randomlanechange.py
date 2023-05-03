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

if __name__ == "__main__":
    startlaneletid = '1412'
    endlaneletid = '1411'
    er = ExiDSampleReader()
    er.read_data()
    root_path = get_package_root_path()
    josmfile = glob.glob(os.path.join(root_path, 
        "assets/exiD/*.osm"))[0]
    recordingMeta_file = glob.glob(os.path.join(root_path, 
        "assets/exiD/*_recordingMeta.csv"))[0]
    recordingMeta = pandas.read_csv(recordingMeta_file)
    utmx, utmy = float(recordingMeta["xUtmOrigin"]), float(recordingMeta["yUtmOrigin"])
    osm = OSMReader(josmfile, utmx, utmy)
    suitable_trackids = get_trackids(er, lambda x, y: y < -250, lambda x, y: x > 550)
    frame_dict = frame_by_frame(er)
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
    # draw the background lanelets
    osm.plot({startlaneletid:'cyan', endlaneletid:'green'}, show_all=False)
    # random track to be shown
    tid = np.random.choice(tracks_with_one_lane_change)
    # now show the track number tid (first figure out the frames to show)
    centerpts, _, frames, _, _, _, _, _, _ = er.get_track(tid)
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
    vehicleobj = None
    def animate(i):
        global vehicleobj
        blit_objs = []
        bbox = None
        for vehicle, fid in frame_dict[frames[startframe+i]]:
            if vehicle == tid:
                bbox = er.bboxes[vehicle][fid]
                break
        fc = 'brown'
        if vehicleobj is not None:
            vehicleobj.set_xy(bbox)
        else:
            vehicleobj = Polygon(bbox, facecolor=fc)
            plt.gca().add_patch(vehicleobj)
        blit_objs = [vehicleobj]
        return blit_objs
    ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate,
        frames=endframe-startframe+1, interval=1, blit=True, repeat=False)
    ani.save("randomsampletrack.gif", writer=matplotlib.animation.ImageMagickWriter(fps=25,
                extra_args=['-loop', '0']),
                progress_callback=lambda i, n: print("%d/%d" % (i, n)))
    # plt.show()