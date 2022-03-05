from cmath import pi
from utils.Pipeline2D import Pipeline2D
from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import numpy as np
import time

# ########### Parameters #############
# data
RESOLUTION = 200
N = 512
DOUBLE_ANGLES = False
DOUBLE_PROJECTIONS = True

ADD_CIRCLE_PADDING = True

#graph
K = 9

#noise
SNR=10

# kernel
# epsilon = 0.5

debug_plot = True
# angle_generation = 'linear_spaced'
angle_generation = 'uniform'

use_wandb = True


#input = shepp_logan_input() # 400, 400

t = time.time()

input = np.load("toyModel/bunny_2d.npy")
pipeline = Pipeline2D(input, RESOLUTION, add_circle_padding=ADD_CIRCLE_PADDING)

pipeline.setup_angles(N, angle_generation=angle_generation, double_angles=DOUBLE_ANGLES)
pipeline.forward(SNR, distances=True, knn=True, K=K, double_projections=DOUBLE_PROJECTIONS)

pipeline.reconstruct_with_graph_laplacian_angles(reconstruction_angles='linear_spaced')

pipeline.log_wandb(execution_time=time.time()-t)