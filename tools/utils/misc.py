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
