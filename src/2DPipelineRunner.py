from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

import time

from utils.Plotting import *
from utils.Graph import *
from utils.Data import find_sigma_noise

import math

def verify_epsilon(distances, epsilon_vdm_list=[50, 5, 1, 0.5, 0.3, 0.1, 0.05, 0.005, 0.0005, 0.00005], show=False, samples=5):
  number_of_plots = len(epsilon_vdm_list)
  dimy = math.ceil(number_of_plots / 2)
  dimx = 2
  fig, axs = plt.subplots(dimy, dimx)

  for i in range(number_of_plots):
    epsilon_vdm = epsilon_vdm_list[i]
    n = distances.shape[0]
    print("n",n)
    step = n//samples
    idx = np.linspace(0, n-step, step ).astype(int)
    x = np.array([ dist ** 2 / epsilon_vdm for dist in distances[idx]])
    y = np.array([ np.exp( - x_i) for x_i in x])
    axs[math.floor(i/2), i % 2].scatter(x,y)
    axs[math.floor(i/2), i % 2].set_title(f"Epsilon VDM {epsilon_vdm}, dist mean: {distances.mean()}")
    axs[math.floor(i/2), i % 2].set_yticks(np.arange(0, 1.1, 0.2))
  if show:
    plt.show()

# noise = np.random.randn(RESOLUTION, N) * np.sqrt(VARIANCE)

def add_noise(SNR,sinogram):
  sinogram=np.array(sinogram)
  VARIANCE=10**(-SNR/10)*(np.std(sinogram)**2)
  noise = np.random.randn(sinogram.shape[0],sinogram.shape[1])*np.sqrt(VARIANCE)
  return sinogram + noise

def sinogramm_l2_distances(sinogram):
  N = sinogram.shape[0]
  dist = np.array([ np.linalg.norm(sinogram[i] - sinogram[j]) for i in range(N) for j in range(N)]).reshape((N,N))
  dist /= dist.max()
  return dist

def estimate_angles(graph_laplacian, degree=False):
  # arctan2 range [-pi, pi]
  angles = np.arctan2(graph_laplacian[:,0],graph_laplacian[:,1]) + np.pi
  # sort idc ascending, [0, 2pi]
  idx  = np.argsort(angles)

  if degree:
    return np.degrees(angles), idx, np.degrees(angles[idx])
  else:
    return angles, idx, angles[idx]

# ########### Parameters #############
# data
RESOLUTION = 200
N = 512
DOUBLE_ANGLES = False
DOUBLE_PROJECTIONS = True
SEED = 2022

ADD_CIRCLE_PADDING = True

#graph
K = 9

#noise
SNR=10

# kernel
epsilon = 0.5

debug_plot = True
color_map = plt.cm.get_cmap('gray')
reversed_color_map = color_map.reversed()

# angle_generation = 'linear_spaced'
angle_generation = 'uniform'

use_wandb = True


################### WAB initialization ###################
if use_wandb:
  import wandb
  
  config = {
    "samples": N,
    "resolution": RESOLUTION,
    "noise_SNR": SNR,
    "k-nn": K,
    "double_angles" : DOUBLE_ANGLES,
    "doube_projections" : DOUBLE_PROJECTIONS,
    "angle_generation" : angle_generation
  }

  # wandb.init(project="2d-pipeline", entity="cedric-mendelin", config=config)


# ############ Loading input image ###############
#input = shepp_logan_input() # 400, 400
input = np.load("toyModel/bunny_2d.npy")
scaleX = RESOLUTION/ input.shape[0]
scaleY = RESOLUTION/ input.shape[1]
input = rescale(input, scale=(scaleX, scaleY), mode='reflect', multichannel=False)


if ADD_CIRCLE_PADDING:
  r = np.ceil(np.sqrt(2 * (RESOLUTION ** 2)))
  padding = int(np.ceil((r - RESOLUTION) / 2))
  print("r:", r)
  print("padding:", padding)
  p = (padding, padding)
  input = np.pad(input, [p,p], mode='constant', constant_values=0)

  RESOLUTION += 2 * padding

# input = input / np.linalg.norm(input)

if debug_plot:
  plot_imshow(input, title='Original input rescaled', c_map=color_map)

# ############### Angles for forward model ############
if angle_generation == 'linear_spaced':
  thetas = np.linspace(0, 2 * np.pi, N)

if angle_generation == 'uniform':
  rng = default_rng(SEED)
  thetas = rng.uniform(0, 2 * np.pi, N)

if DOUBLE_ANGLES:
  thetas = np.concatenate((thetas, np.mod(thetas + np.pi , 2 * np.pi)))
  N = N * 2


# radon transform from skimage works with degrees, not radians.
thetas_degree = np.degrees(thetas)

# plot_2dscatter(np.cos(thetas_degree), np.sin(thetas_degree), title=f"Input angles")

# ############### Forward model (Radon Transform) ###########
Rf = radon(input, theta=thetas_degree, circle=True)
sinogram_data = Rf.transpose()

if DOUBLE_PROJECTIONS:
  thetas = np.concatenate((thetas, np.mod(thetas + np.pi , 2 * np.pi)))
  # thetas = np.concatenate((thetas, thetas + np.pi))
  thetas_degree = np.degrees(thetas)
  sinogram_data = np.concatenate((sinogram_data, np.flip(sinogram_data, 1)))
  N = 2 * N

# Add some noise to this projection.
sinogram_data_noisy = add_noise(SNR, sinogram_data)

if debug_plot:
  plot_imshow(sinogram_data[np.argsort(thetas_degree)], title='Sinogram original input', aspect='auto', c_map=reversed_color_map)
  plot_imshow(sinogram_data_noisy[np.argsort(thetas_degree)], title='Sinogram noisy', aspect='auto', c_map=reversed_color_map)

##################### Define distances ##########################
# clean case
sinogram_dist = sinogramm_l2_distances(sinogram_data)
#dist = np.exp(- dist ** 2 /  epsilon)

# for k in range(2,12,1):
#   graph, classes = generate_knn_from_distances(sinogram_dist, K=k, ordering='asc', ignoreFirst=True)
#   graph_laplacian = calc_graph_laplacian(graph, embedDim=2)
#   plot_2d_scatter(graph_laplacian, title=f"Graph Laplacian sinogram graph K = {k}")
graph, classes = generate_knn_from_distances(sinogram_dist, K=K, ordering='asc', ignoreFirst=True)
graph_laplacian = calc_graph_laplacian(graph, embedDim=2)
plot_2d_scatter(graph_laplacian, title=f"Graph Laplacian sinogram graph K = {K}")


# noisy case
sinogram_noisy_dist = sinogramm_l2_distances(sinogram_data)
graph_noisy, classes_noisy = generate_knn_from_distances(sinogram_noisy_dist, K, ordering='asc', ignoreFirst=True)

graph_laplacian_noisy = calc_graph_laplacian(graph_noisy, embedDim=2)
plot_2d_scatter(graph_laplacian_noisy, title=f"Graph Laplacian noisy sinogram graph")

# ######################## Angle estimation #####################
angles_estimated, angles_estimated_indices, angles_estimated_sorted  = estimate_angles(graph_laplacian, degree=True)
noisy_angles_estimated, noisy_angles_estimated_indices, noisy_angles_estimated_sorted  = estimate_angles(graph_laplacian_noisy, degree=True)

print("Diff sinogram angles:", np.linalg.norm(thetas_degree - angles_estimated))
print("Diff noisy sinogram angles:", np.linalg.norm(thetas_degree - noisy_angles_estimated))

# print("angles estimated:", angles_estimated_sorted)
# print("noisy angles estimated", noisy_angles_estimated_sorted)

########################## Backward model ###########################
# angle assumption, equally spaced:
angles = np.linspace(0, 360, N)

# clean case
sinogram_data_sorted = sinogram_data[angles_estimated_indices]
out = iradon(sinogram_data_sorted.T, theta=angles,circle=True)
plot_imshow(out, title='Original input reconstructed estimated angles', c_map=color_map)


# noisy case
sinogram_data_noisy_sorted = sinogram_data_noisy[noisy_angles_estimated_indices]
out2 = iradon(sinogram_data_noisy_sorted.T, theta=angles,circle=True)
plot_imshow(out2, title='Noisy input reconstructed', c_map=color_map)




plt.show()


# class Pipeline2DCT():
#   def __init__()
