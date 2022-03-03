

import numpy as np
import logging


from aspire.volume import Volume

from utils.Data import *
from utils.AspireHelpers import *
from utils.Plotting import *
import time
import mrcfile

from utils.Plotting import plot_voxels
from utils.Geometry import *
from utils.Cheng import getChengBunny, getVedoBunny, getMapFile
from utils.Vedo import visualize_voxels_3d
from vedo import dataurl, Mesh, mesh2Volume, volumeFromMesh, signedDistanceFromPointCloud
logger = logging.getLogger(__name__)

#aspire_vol = Volume(V)


img_size = 50  # image size in square
n_img = 1000  # number of images
snr = 10
K = 10

# mesh = Mesh(dataurl + "bunny.obj").normalize().subdivide()
# mesh.points
# vol = mesh2Volume(mesh, spacing=(0.01,0.01,0.01)).tonumpy()
# V = normalize_min_max(downsample_voxels(vol, img_size))

# V = getMapFile(resolution=img_size)
V = getChengBunny(img_size)


#visualize_voxels_3d(V)


#plot_imshow(V[:,:,20])


aspire_vol, sim, clean_graph, noisy_graph = reconstruct_result_cheng(V, n_img, img_size, snr, K)


# sim.images(0, 10).show()
# sim.clean_images(0, 10).show()


# input("enter to continue")

A = create_adj_mat_nx(clean_graph.classes)

embedding = calc_graph_laplacian(A, numberOfEvecs=3)
embedding = normalize_range(embedding, -1, 1)
# plot_3d_scatter(embedding)

embedding_aligned = align_3d_embedding_to_shpere(embedding, debug=True)
rots, angles  = calc_rotation_from_points_on_sphere_ZYZ(embedding_aligned, debug=True)

aspire_angles_xyz = np.array([R.from_euler("ZYZ", angles).as_euler("XYZ") for angles in sim.angles])
reconstruction_rots = np.zeros((n_img, 3))

for i in range(n_img):
    angle = sim.angles[i]
    aspire_rot = aspire_angles_xyz[i]
    inplane_rotation = aspire_rot[0]

    #rot = R.__mul__(R.from_matrix(rot_matrix_z(inplane_rotation)), R.from_euler("ZXZ" , rots[i]) )
    reconstruction_rots[i] = np.array([inplane_rotation, rots[i][1], rots[i][2] ])

    if i % 100 == 0:
        rot1 = rot.as_matrix()
        rot2 = R.from_euler("XYZ",aspire_angles_xyz[i]).as_matrix()
        r = np.dot(rot1, rot2.T)
        theta = (np.trace(r) - 1) / 2
        diff = np.arccos(theta)
        print(f"reconstruction: {reconstruction_rots[i]}, true: {aspire_angles_xyz[i]} , inplane_rot: {inplane_rotation}, estimated zx rot: {rots[i]}, diff: {diff}")


#print(rots)
#print(sim.angles)


# for i in range(n_img):
#     if i % 100 == 0:
#         rot1 = R.from_euler("ZYZ", sim.angles[i]).as_matrix()
#         rot2 = R.from_euler("ZYZ", rots[i]).as_matrix()
#         r = np.dot(rot1, rot2.T)
#         theta = (np.trace(r) - 1) / 2
#         diff = np.arccos(theta)
#         print(f"euler_given: {sim.angles[i]}, euler_estimated: {rots[i]}, diff: {diff}")




clean_images = sim.clean_images(0, n_img).asnumpy()

#func = lambda angles:  R.from_euler(angles, "ZYZ").as_euler("xyz")




rec_given_rots = reconstruction_naive(clean_images, n_img, img_size, aspire_angles_xyz)
plot_imshow(rec_given_rots[:,:,25])
voxelSaveAsMap(rec_given_rots, 'rec_given_rots_chengBunny.map')

rec_given_rots = reconstruction_naive(clean_images, n_img, img_size, reconstruction_rots)
plot_imshow(rec_given_rots[:,:,25])
voxelSaveAsMap(rec_given_rots, 'rec_estimated_rots_chengBunny.map')




# rec_calculated_rots = reconstruction_naive(clean_images, n_img, img_size, rots)
# plot_imshow(rec_calculated_rots[:,:,25])
# voxelSaveAsMap(rec_calculated_rots, 'rec_calculated_rots_myBunny.map')
assert False

rot_using = np.array([alpha_beta[:,0], alpha_beta[:,1], sim.angles[:,2]]).T
print(rot_using.shape)







A_noisy = create_adj_mat_nx(noisy_graph.classes)
embedding_noisy = calc_graph_laplacian(A_noisy, numberOfEvecs=3)
embedding_noisy = normalize_range(embedding_noisy, -1, 1)

plot_3dscatter(embedding_noisy[:, 0], embedding_noisy[:, 1], embedding_noisy[:, 2],  (10,10))


embedding_noisy_mapped = align_3d_embedding_to_shpere(embedding_noisy)

rots, alpha_beta = calc_rotation_from_points_on_sphere(embedding_noisy_mapped)

noisy_images = sim.images(0, n_img).asnumpy()

rec_calculated_rots = reconstruction_naive(noisy_images, n_img, img_size, rots)
rec_given_rots = reconstruction_naive(noisy_images, n_img, img_size, sim.angles)

voxelSaveAsMap(rec_calculated_rots, 'rec_noisy_calculated_rots_2.map')
voxelSaveAsMap(rec_given_rots, 'rec_noisy_given_rots_2.map')