import numpy as np


def set_small_values_to(x, new_val=0, threshold=1e-13):
    x[np.abs(x) < threshold] = new_val
    return x

def normalize_range(x, lower, upper):
    x1 = (upper - lower) * ((x - np.min(x)) / np.ptp(x))  + lower
    return x1

def normalize_min_max(x):
    x1 = (x - np.min(x)) / np.ptp(x) # ptp: max-min
    return x1

def set_negatives_and_small_values_to(x, new_val=0, threshold=1e-13):
    x[x < threshold] = new_val
    return x

def set_small_values_to(x, new_val=0, threshold=1e-13):
    x[np.abs(x) < threshold] = new_val
    return x


def normalize_cheng(x):
    x = x - (x.max() + x.min()) / 2
    x=x/x.max()
    return x