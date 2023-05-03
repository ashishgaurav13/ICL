import warnings
import datetime
import copy
import numpy as np
from inspect import getfile
import os
import torch
import tools
import numba
from scipy.spatial.distance import cdist
import ot
import matplotlib.pyplot as plt

def combine_dicts(*args):
    """
    Combine multiple dictionaries.
    Taken from github.com/ashishgaurav13/wm2
    """
    ret = {}
    for d in args:
        for key, value in d.items():
            if key in ret.keys():
                if type(ret[key]) == set:
                    if type(value) in [list, set]:
                        ret[key].update(value)
                    else:
                        ret[key].add(value)
                else:
                    if type(ret[key]) in [list, set]:
                        ret[key] = set([*ret[key]])
                    else:
                        ret[key] = set([ret[key]])
                    if type(value) in [list, set]:
                        ret[key].update(value)
                    else:
                        ret[key].add(value)
            else:
                ret[key] = value
    return ret

def dict_to_numpy(x):
    """
    Convert dictionary to a numpy array, recursively.
    Any None data will return a -1.0 in its place.
    Taken from github.com/ashishgaurav13/wm2
    """
    if type(x) != dict:
        if x is None: return -1.0
        return x
    if x == {}: return np.array([])
    return np.hstack([dict_to_numpy(val) for val in x.values()]).flatten()

def nowarnings():
    """
    Ignore all warnings.
    """
    warnings.filterwarnings("ignore")

def timestamp():
    """
    Return the current timestamp as a str.
    """
    return datetime.datetime.now().strftime("%d-%h-%Y-%I-%M-%S-%p")

def rewards_to_returns(R, discount_factor):
    """
    Create returns [G0, G1, ...]
    Gn = Rn + discount_factor * R{n+1} + ...
    """
    has_key = "discount_matrix" in tools.store.data.keys()
    diff_discount = has_key and \
        tools.store.data["discount_matrix"][-1] != discount_factor
    bigger_matrix = has_key and \
        tools.store.data["discount_matrix"][-1] == discount_factor and \
        len(R) > tools.store.data["discount_matrix"][0].shape[0]
    if (not has_key) or diff_discount or bigger_matrix:
            D = [discount_factor**i for i in range(len(R))]
            tools.store.data["discount_matrix"] = (
                torch.Tensor([\
                    [0,]*i+D[:len(R)-i] \
                    for i in range(len(R))]),
                discount_factor
            )
    discounted_rewards = torch.Tensor(R)
    discounted_rewards = tools.store.data["discount_matrix"][0][:len(R), :len(R)]\
        @ discounted_rewards
    return discounted_rewards

def demo():
    """
    Demo function. Does nothing.
    """
    pass

def get_package_root_path():
    """
    Get root path of the package.
    """
    demo_path = getfile(demo)
    return os.path.abspath(os.path.join(demo_path, "../.."))

def less_all(item, lst):
    """
    Return True if item is strictly less than all elements in given lst.
    """
    for comp in lst:
        if item >= comp:
            return False
    return True

def less_equal_all(item, lst):
    """
    Return True if item is strictly less than or equal to all elements
    in given lst.
    """
    for comp in lst:
        if item > comp:
            return False
    return True

def greater_all(item, lst):
    """
    Return True if item is strictly greater than all elements in given lst.
    """
    for comp in lst:
        if item <= comp:
            return False
    return True

def greater_equal_all(item, lst):
    """
    Return True if item is strictly greater than or equal to all elements
    in given lst.
    """
    for comp in lst:
        if item < comp:
            return False
    return True

def convert_lambdas(d):
    """
    Convert lambdas in a given dictionary/list.
    """
    if type(d) == str:
        if "tools" in d or "lambda" in d: 
            return eval(d)
        return d
    if type(d) == list:
        ret_list = []
        for item in d:
            ret_list += [convert_lambdas(item)]
        return ret_list
    if type(d) == dict:
        ret_dict = {}
        for key, value in d.items():
            ret_dict[key] = convert_lambdas(value)
        return ret_dict
    return d

def wasserstein_distance2d(u, v, p='cityblock'):
    """
    Wasserstein distance in 2D
    stackoverflow.com/questions/57562613/python-earth-mover-distance-of-2d-arrays
    """
    u = np.array(u)
    v = np.array(v)
    assert(u.shape == v.shape and len(u.shape) == 2)
    dim1, dim2 = u.shape
    assert(p in ['euclidean', 'cityblock'])
    coords = np.zeros((dim1*dim2, 2)).astype('float')
    for i in range(dim1):
        for j in range(dim2):
            coords[i*dim2+j, :] = [i, j]
    d = cdist(coords, coords, p)
    u /= u.sum()
    v /= v.sum()
    return ot.emd2(u.flatten(), v.flatten(), d)

def mse(u, v):
    """
    Mean squared error.
    """
    u = np.array(u)
    v = np.array(v)
    assert(u.shape == v.shape)
    return np.mean(np.power(u-v, 2))

### https://kite.com/python/answers/how-to-check-if-two-line-segments-intersect-in-python
def on_segment(p, q, r):
    if (r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and
        r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1])):
        return True
    return False

def orientation(p, q, r):
    val = (((q[1] - p[1]) * (r[0] - q[0])) -
           ((q[0] - p[0]) * (r[1] - q[1])))
    if val == 0:
        return 0
    return 1 if val > 0 else -1

def intersects(seg1, seg2):
    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and on_segment(p1, q1, p2): return True
    if o2 == 0 and on_segment(p1, q1, q2): return True
    if o3 == 0 and on_segment(p2, q2, p1): return True
    if o4 == 0 and on_segment(p2, q2, q1): return True

    return False

###
def in_regions(prev_state, next_state, regions):
        """Returns True if agent moves through/on rectangles defined
        by `regions'."""
        for region in regions:
            if in_rectangle(prev_state, region):
                return True
            if in_rectangle(next_state, region):
                return True
            for bound in boundaries(*region):
                if intersects((prev_state, next_state), bound):
                    return True
        return False

def boundaries(o, w, h):
    """Returns the boundaries of rectangle of width w and height h with the
    bottom left corner at the point o.
    """
    return [(o, o + np.array([w,0])),
            (o, o + np.array([0,h])),
            (o + np.array([w,0]), o + np.array([w, h])),
            (o + np.array([0,h]), o + np.array([w, h]))]

def in_rectangle(state, region):
    """Returns True if a state ((x,y) coordinate) is in a rectangle defined
    by region (a tuple of origin, width and height).
    """
    o, w, h = region
    if (state[0] > o[0] and state[0] < o[0] + w and
        state[1] > o[1] and state[1] < o[1] + h):
        return True

def add_circle(ax, point, color, radius=0.2, clip_on=False):
    circle = plt.Circle(
            point,
            radius=radius,
            color=color,
            clip_on=clip_on
    )
    ax.add_patch(circle)

def figure_to_array(fig):
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


