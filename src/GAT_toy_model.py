import imageio
from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon

from utils.SingleImagePipeline import SingleImagePipeline
from utils.Graph import *
import matplotlib.pyplot as plt

import torch
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from scipy.spatial import distance_matrix
import torch.nn.functional as F
import math

import numpy as np
from numpy.random import default_rng

##################### Helpers ################
def add_noise(SNR, sinogram):
    sinogram=np.array(sinogram)
    VARIANCE=10**(-SNR/10)*(np.std(sinogram)**2)
    noise = np.random.randn(sinogram.shape[0],sinogram.shape[1])*np.sqrt(VARIANCE)
    return sinogram + noise

def estimate_angles(graph_laplacian, degree=False):
    # arctan2 range [-pi, pi]
    angles = np.arctan2(graph_laplacian[:,0],graph_laplacian[:,1]) + np.pi
    # sort idc ascending, [0, 2pi]
    idx  = np.argsort(angles)

    if degree:
      return np.degrees(angles), idx, np.degrees(angles[idx])
    else:
      return angles, idx, angles[idx]


################## Parameters ########################
RESOLUTION = 200
N = 512
DOUBLE_ANGLES = True

SEED = 2022

ADD_CIRCLE_PADDING = True

#graph
K = 9

#noise
SNR=25


input = imageio.imread('src/maps/toy_image.png')
input = shepp_logan_phantom()

########################### Setup angles #########################
rng = default_rng(SEED)
input_angles = rng.uniform(0, 2 * np.pi, N)
input_angles = np.concatenate((input_angles, np.mod(input_angles + np.pi , 2 * np.pi)))
N = N * 2
input_angles_degrees = np.degrees(input_angles)

reconstruction_angles_degrees =  np.linspace(0, 360, N)
reconstruction_angles =  np.linspace(0, 2 * np.pi, N)

##################### forward ########################
Rf = radon(input, theta=input_angles_degrees, circle=True)
sinogram = Rf.T
noisy_sinogram = add_noise(SNR, sinogram)

# distances:
distances = distance_matrix(sinogram, sinogram)
distances /= distances.max()

noisy_distances = distance_matrix(noisy_sinogram, noisy_sinogram)
noisy_distances /= noisy_distances.max()

# k-nn:
graph, classes, edges = generate_knn_from_distances_with_edges(distances, K, ordering='asc', ignoreFirst=True)
noisy_graph, noisy_classes, noisy_edges = generate_knn_from_distances_with_edges(noisy_distances, K, ordering='asc', ignoreFirst=True)

# gl:
graph_laplacian = calc_graph_laplacian(graph, embedDim=2)
noisy_graph_laplacian = calc_graph_laplacian(noisy_graph, embedDim=2)

################### GAT class #############################
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_layer_dimension=16, num_classes=2):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_layer_dimension)
        self.conv2 = GATConv(hidden_layer_dimension, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        #return F.log_softmax(x, dim=1)
        return x

class GAT_angles(torch.nn.Module):
    def __init__(self, num_node_features, hidden_layer_dimension=16, num_classes=2):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_layer_dimension)
        self.conv2 = GATConv(hidden_layer_dimension, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        #x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        #return F.log_softmax(x, dim=1)
        return x

################ GAT #################
# prep edges:
N_noisy_edges = len(noisy_edges)
edge_index = np.zeros((2, N_noisy_edges))
edge_attribute = np.zeros((N_noisy_edges, 1))
for i in range(N_noisy_edges):
  (n,m) = noisy_edges[i]
  edge_index[0,i] = n
  edge_index[1,i] = m
  edge_attribute[i] = noisy_distances[n,m]

plt.show()