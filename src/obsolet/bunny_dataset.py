from utils.Data import *

import matplotlib.pyplot as plt
import utils.Plotting as plot

plt.ion()
import numpy as np
import time
import wandb

import sys
sys.path.insert(0, '..')

################### Parameters ###################
N = 100
#dim = 3
image_res = 100

# Noise
snr = 10

# KNN
K=6

################### WAB initialization ###################
config = {
  "samples": N,
  "sample size": (100, 100),
  "noise_snr": snr,
  "knn_K": K
}

wandb.init(project="bunny-dataset", entity="cedric-mendelin", config=config)



################ Data loading ##############

t = time.time()


original_images, uniform_3d_angles, noisy_images, noise = create_or_load_bunny_immages(N, image_res, snr, save=True)

plot.plot_imshow(noisy_images[1])

wandb.log({"noisy_image_example": wandb.Image(noisy_images[0])})
wandb.log({"time_datageneration": time.time()-t})

############### KNN clean graph ###################
from utils.Graph import *
from utils.FourierBessel import *

t = time.time()

#fb = FBBasis2D((image_res,image_res),ell_max=5,dtype=np.float64)
#dist, ind, angles = fb.Knn_mat_reduce_ram(original_images, K, M, False)
classes, reflections, rotations, correlations = create_or_load_knn_rotation_invariant(name='bunny', N=N, image_res=image_res, images=original_images, K=K)
classes_noisy, _, _, _ = create_or_load_knn_rotation_invariant(name='bunny_noisy', N=N, image_res=image_res, images=noisy_images, K=K, snr=snr)

wandb.log({"time_knn": time.time()-t})

#np.savez_compressed('aspire_knn_original_500_10.np', classes, reflections, rotations, correlations)
#np.savez_compressed('aspire_knn_original_500_10.np', dist=dist, indices=ind, angles=angles)
#np.savez_compressed('original_images_500_100.np', original_images, uniform_3d_angles)


############### Laplacian ###################

t = time.time()
embedding = calc_graph_laplacian_from_knn_indices(classes, numberOfEvecs=3)

wandb.log({"time_laplacian": time.time()-t})


plot.plot_3dscatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],  (10,10))


wandb.log({"plot_laplacian": wandb.Object3D(embedding)})

wandb.finish()

input("Enter to terminate")