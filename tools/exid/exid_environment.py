import glob
import os
import numpy as np
from tools.utils import get_package_root_path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation
import matplotlib.pyplot as plt
from .exid_trafficanimation import ExiDSampleReader, get_trackids, frame_by_frame

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