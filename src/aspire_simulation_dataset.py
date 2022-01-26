import numpy as np
import logging


from aspire.volume import Volume

from utils.Data import *
from utils.AspireHelpers import *
from utils.Plotting import *
import time
import mrcfile

from utils.Plotting import plot_voxels

logger = logging.getLogger(__name__)

# aspire_vol = vedo_bunny_to_asipre_volume()

DATA_DIR = "src/maps/"  # Tutorial example data folder
map_name = 'bunny.map'

# Main:
t = time.time()


v_npy = mrcfile.open(DATA_DIR + map_name ).data.astype(np.float32)

aspire_vol = Volume(v_npy)


img_size = 100  # image size in square
n_img = 1000  # number of images
snr = 10
K = 10

aspire_vol, sim, clean_graph, noisy_graph = create_or_load_dataset_from_map("bunny-test", map_name, n_img, img_size, snr, K)

A = create_adj_mat_nx(clean_graph.classes)

embedding = calc_graph_laplacian(A, numberOfEvecs=3)
embedding_normalized = normalize_min_max(embedding)

plot_3dscatter(embedding_normalized[:,0], embedding_normalized[:,1], embedding_normalized[:,2])

embedding_normalized = align_3d_embedding_to_shpere(embedding_normalized)

plot_3dscatter(embedding_normalized[:,0], embedding_normalized[:,1], embedding_normalized[:,2])



A_noisy = create_adj_mat_nx(noisy_graph.classes)
embedding_noisy = calc_graph_laplacian(A_noisy, numberOfEvecs=3)

print(embedding_noisy)


plot_3dscatter(embedding_noisy[:, 0], embedding_noisy[:, 1], embedding_noisy[:, 2],  (10,10))



diff = embedding - embedding_noisy

assert np.all(diff != 0) , "embeddings are same"

input("enter to terminate")


print(time.time() - t)

assert False

## prepare simulation

aspire_vol = aspire_vol.downsample(img_size)
#plot_voxels(aspire_vol.__getitem__(0))

# %%
# Defining rotations

rot_angles = random_rotation_3d(n_img)
noise_variance = find_sigma_noise(snr, aspire_vol)


src = create_simulation(aspire_vol, n_img, rot_angles, noise_variance)

logger.info(f"Simulation creation = {time.time() -t}")

t = time.time()

#src.save("out.star", save_mode="single", overwrite=True)
#src.save_metadata("out2.star")
# src.save_images("out3.star", overwrite=True, filename_indices=["out.mrcs"])

# %%
# Yield projection images from the Simulation Source
# --------------------------------------------------

# Consume images from the source by providing
# a starting index and number of images.
# Here we generate the first 3 and peek at them.
#src.images(0, 3).show()
#src.projections(0, 3).show()


# Here we return the first n_img images as a numpy array.

t = time.time()


#t = src.get_metadata("__filter_indices")
#print(aspire_vol.dtype)
#t = dirty_ary[0].get_metadata("__filter_indices")
#print(t)


# And we have access to the clean images
clean_ary = src.images(0, n_img).asnumpy()

dist_knn, idx_best_img_knn, angle_est_knn, refl_est_knn = rotation_invariant_knn(clean_ary, K=K)

logger.info(f"Clean Graph creation = {time.time() -t}")

t = time.time()

dirty_ary = src.projections(0, n_img).asnumpy()
dist_knn_noisy, idx_best_img_knn_noisy, angle_est_knn_noisy, refl_est_knn_noisy = rotation_invariant_knn(dirty_ary, K=K)

logger.info(f"Noisy Graph creation = {time.time() -t}")

#### Reconstruction #####
rec_vol = reconstruction_naive(clean_ary, n_img, img_size, rot_angles)


print(rec_vol.max())
print(rec_vol.min())
print(rec_vol.mean())

rec_vol = set_negatives_and_small_values_to(rec_vol, threshold=0.45)

plot_voxels(rec_vol)


diff = aspire_vol.__getitem__(0) - rec_vol

print(diff)



input("enter to terminate")

#plot_voxels(rec_vol)