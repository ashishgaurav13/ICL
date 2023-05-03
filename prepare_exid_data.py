from tools.exid import ExiDSampleReader, frame_by_frame
import pandas
from tools.exid import OSMReader
from tools.exid import load_trajectories2, lane_change_start_end2, \
    find_closest_node, find_segment
from shapely.geometry import Point, LineString
import glob, os, pandas

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
    # for item in lane_change_data:
    #     print(item)