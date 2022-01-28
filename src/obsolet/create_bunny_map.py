import matplotlib
from vedo import dataurl, Mesh, mesh2Volume
from aspire.volume import Volume
import numpy as np
import matplotlib.pyplot as plt
from utils.Data import voxelSaveAsMap

resolution = 50

mesh = Mesh(dataurl + "bunny.obj").normalize().subdivide()
mesh.points
vol = mesh2Volume(mesh, spacing=(0.01,0.01,0.01))

vol_equally_spaced = np.pad(vol.tonumpy(), ((0,3),(0,6),(0,57)), mode='constant', constant_values=0).astype(np.float32)

print(vol_equally_spaced.shape)

V = Volume(vol_equally_spaced).__getitem__(0)

print(V.max())
print(V.min())
print(V.mean())


voxelSaveAsMap(V)



