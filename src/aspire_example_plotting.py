from aspire.volume import Volume

import vedo

import logging
import os

import mrcfile
import numpy as np

import aspire
import matplotlib.pyplot as plt

DATA_DIR = "src/maps"  # Tutorial example data folder
#v_npy = mrcfile.open('C:\master-thesis\src\maps\emd_25792.map').data.astype(np.float32)
map_name = 'out.map'
v_npy = mrcfile.open('C:\master-thesis\\' + map_name ).data.astype(np.float32)

print(v_npy.shape)
# Then using that to instantiate a Volume, which is downsampled to 60x60x60
#v = Volume(v_npy).downsample(100)


#res = np.array(v.asnumpy())

#print(res[0].shape)
#print(v.asnumpy().shape)
#print(v.to_vec().shape)

vol = vedo.Volume(v_npy)

vol.show()

#print(v.__getitem__(0).reshape((1125,3)).shape)

#from vedo import *

#pts = Points(v.__getitem__(0).reshape((1125, 3)))
#pts.show()

#reco = recoSurface(pts, dims=15)
#reco.show()