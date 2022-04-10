import numpy as np
from utils.Plotting import *
from utils.Geometry import *
from utils.obsolete.Data import voxelSaveAsMap
from scipy.spatial.transform import Rotation as R
from numpy.random import default_rng
from scipy.ndimage.interpolation import rotate
from skimage.transform import resize
import mrcfile

from vedo import dataurl, Mesh, mesh2Volume, volumeFromMesh


mesh = Mesh(dataurl + "bunny.obj").normalize().subdivide()

resolution = 240

padding = 8

vol = mesh2Volume(mesh, spacing=(0.01,0.01,0.01)).tonumpy()
V = normalize_min_max(downsample_voxels(vol, resolution - (2*padding)))
p = (padding, padding)
V = np.pad(V, (p, p, p) , mode='constant', constant_values=0).astype(np.float64)


print(V.max()) 
print(V.min())
print(V.mean()) # 48.72

voxelSaveAsMap(V, "padding_bunny.map")