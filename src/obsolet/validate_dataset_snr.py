from math import degrees
import numpy as np
from utils.Geometry import *
from utils.Graph import *
from utils.AspireHelpers import *
from utils.obsolete.Data import *
from utils.Plotting import *
import os.path as path
from os import mkdir
import gemmi
import vedo
import mrcfile

from aspire.volume import Volume
DATA_DIR = 'src/data/'
MAP_DIR = 'src/maps/' 


img_size = 64  # image size in square
n_img = 1000  # number of images
snr = 10
k = 10
normalize = True


map_name = 'bunny.map'

# folders and files
    
map_file_path = MAP_DIR + map_name

# load map file
v_npy = mrcfile.open(map_file_path ).data.astype(np.float32)

# normalize
if normalize:
    v_npy = normalize_min_max(v_npy)

# create aspire volume and downsample to image size
aspire_vol = Volume(v_npy)
aspire_vol = aspire_vol.downsample(img_size)

# return values:
rotation_angles = rotation_angles = random_rotation_3d(n_img)

# determine noise
noise_variance = find_sigma_noise(snr, aspire_vol)
#found noise_variance: 0.12354835437561838
print(f"found noise_variance: {noise_variance}")

noise_variance = find_sigma_noise_cheng(snr, aspire_vol)


#found noise_variance: 0.12354835437561838
print(f"found noise_variance: {noise_variance}")

#noise_variance = 1e-10

# create aspire simulation
sim = create_simulation(aspire_vol, n_img, rotation_angles, noise_variance, ctf=np.zeros((7)))

sim.projections(0,4).show()
sim.images(0,4).show()


rot_difference = np.zeros(n_img)
for i in range(n_img):
    rot1 = R.from_euler(my_euler_sequence(), rotation_angles[i], degrees=False).as_matrix()
    rot2 = R.from_euler(my_euler_sequence(), sim.angles[i], degrees=False).as_matrix()
    rot_result = np.dot(rot1, rot2.T)
    theta = (np.trace(rot_result) -1)/2
    rot_difference[i] = np.arccos(theta) * (180/np.pi)


print(f"rotations:{rotation_angles}")
print(f"sim rotations:{sim.angles}")

print(f"diff std: {np.nan_to_num(rot_difference).sum() / n_img}")
    



# get clean graph
images = sim.projections(0, n_img)
distance, classes, angles,reflection = rotation_invariant_knn(images, K=k)


sim.vol_coords

A = create_adj_mat_nx(classes)

embedding = calc_graph_laplacian(A, numberOfEvecs=3)

embedding = normalize_range(embedding, -1, 1)

#embedding_normalized = normalize_min_max(embedding)

plot_3dscatter(embedding[:,0], embedding[:,1], embedding[:,2])

embedding_aligned = align_3d_embedding_to_shpere(embedding, debug=True)


rots , _ = calc_rotation_from_points_on_sphere(embedding_aligned)
rots_2 , _  = calc_rotation_from_points_on_sphere_2(embedding_aligned)




rot_difference = np.zeros(n_img)
for i in range(n_img):
    rot1 = R.from_euler(my_euler_sequence(), rots[i], degrees=False).as_matrix()
    rot2 = R.from_euler(my_euler_sequence(), sim.angles[i], degrees=False).as_matrix()
    rot_result = np.dot(rot1, rot2.T)
    theta = (np.trace(rot_result) -1)/2
    rot_difference[i] = np.arccos(theta) * (180/np.pi)

print(f"rotations:{rotation_angles}")
print(f"sim rotations:{sim.angles}")

print(f"diff std: {np.nan_to_num(rot_difference).sum() / n_img}")


rot_difference = np.zeros(n_img)
for i in range(n_img):
    rot1 = R.from_euler(my_euler_sequence(), rots_2[i], degrees=False).as_matrix()
    rot2 = R.from_euler(my_euler_sequence(), sim.angles[i], degrees=False).as_matrix()
    rot_result = np.dot(rot1, rot2.T)
    theta = (np.trace(rot_result) -1)/2
    rot_difference[i] = np.arccos(theta) * (180/np.pi)

print(f"rotations:{rotation_angles}")
print(f"sim rotations:{sim.angles}")

print(f"diff std: {np.nan_to_num(rot_difference).sum() / n_img}")

# get noisy graph
#noisy_graph = create_or_load_knn(noisy_file_path, sim.images(0, n).asnumpy(), k)



input("Enter to terminate")
