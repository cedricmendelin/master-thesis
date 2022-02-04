

import numpy as np
import logging


from aspire.volume import Volume

from utils.Data import *
from utils.AspireHelpers import *
from utils.Plotting import *
from utils.Vedo import visualize_voxels_3d
import time
import mrcfile

from utils.Plotting import plot_voxels
from utils.Geometry import *
from vedo import dataurl, Mesh, mesh2Volume

logger = logging.getLogger(__name__)

#aspire_vol = Volume(V)


img_size = 50  # image size in square
n_img = 10  # number of images
snr = 10
K = 2

mesh = Mesh(dataurl + "bunny.obj").normalize().subdivide()
mesh.points

vol = mesh2Volume(mesh, spacing=(0.01,0.01,0.01)).tonumpy()
vol = normalize_min_max(vol)

#237, 234, 183
padding = (20,20)
vol = np.pad(vol, (padding, padding, padding) , mode='constant', constant_values=0).astype(np.float64)



V = normalize_min_max(downsample_voxels(vol, img_size))
visualize_voxels_3d(V)
plot_imshow(V[:,:,20])

aspire_vol, sim, clean_graph, noisy_graph = reconstruct_result_cheng(V, n_img, img_size, snr, K)


sim.images(0, n_img).show()

sim.projections(0, n_img).show()


input("enter to continue")

A = create_adj_mat_nx(clean_graph.classes)

embedding = calc_graph_laplacian(A, numberOfEvecs=3)
embedding = normalize_range(embedding, -1, 1)
plot_3d_scatter(embedding)

embedding_aligned = align_3d_embedding_to_shpere(embedding, debug=True)
rots  = calc_rotation_from_points_on_sphere_ZYZ(embedding_aligned, debug=True)

#print(rots)
#print(sim.angles)


for i in range(n_img):
    if i % 100 == 0:
        rot1 = R.from_euler("ZYZ", sim.angles[i]).as_matrix()
        rot2 = R.from_euler("ZYZ", rots[i]).as_matrix()
        r = np.dot(rot1, rot2.T)
        theta = (np.trace(r) - 1) / 2
        diff = np.arccos(theta)
        print(f"euler_given: {sim.angles[i]}, euler_estimated: {rots[i]}, diff: {diff}")

        rots[i][0] = sim.angles[i][0]
        rot1 = R.from_euler("ZYZ", sim.angles[i]).as_matrix()
        rot2 = R.from_euler("ZYZ", rots[i]).as_matrix()
        r = np.dot(rot1, rot2.T)
        theta = (np.trace(r) - 1) / 2
        diff = np.arccos(theta)
        print(f"2nd try: euler_given: {sim.angles[i]}, euler_estimated: {rots[i]}, diff: {diff}")

 