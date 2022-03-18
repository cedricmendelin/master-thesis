import imageio
from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon

from utils.SingleImagePipeline import SingleImagePipeline
from utils.Graph import *
from utils.Plotting import *
from utils.pytorch_radon.radon import *
import matplotlib.pyplot as plt

import torch
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from scipy.spatial import distance_matrix
import torch.nn.functional as F
import math

import numpy as np
from numpy.random import default_rng

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##################### Helpers ################
def add_noise(SNR, sinogram):
    
    VARIANCE = 10 ** (-SNR/10) * (torch.std(sinogram) **2) 
    noise = torch.randn(sinogram.shape[0], sinogram.shape[1]) * torch.sqrt(VARIANCE)
    return sinogram + noise

def estimate_angles(graph_laplacian, degree=False):
    # arctan2 range [-pi, pi]
    angles = torch.atan2(graph_laplacian[:,0],graph_laplacian[:,1]) + torch.pi
    # sort idc ascending, [0, 2pi]
    idx  = torch.argsort(angles)

    if degree:
      return torch.rad2deg(angles), idx, torch.rad2deg(angles[idx])
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
input = torch.tensor(shepp_logan_phantom()).type(torch.float)

plot_imshow(input, title="Input image")

########################### Setup angles #########################
# rng = default_rng(SEED)
# input_angles = rng.uniform(0, 2 * np.pi, N)
from torch.distributions.uniform import Uniform
normal = Uniform(torch.tensor([0.0]), torch.pi)
input_angles = normal.sample((N,))
# input_angles = torch.cat((input_angles, torch.remainder(input_angles + torch.pi , 2 * torch.pi)))
# N = N * 2
input_angles_degrees = torch.rad2deg(input_angles).type(torch.float)

reconstruction_angles_degrees =  torch.linspace(0, 180, N).type(torch.float)
reconstruction_angles =  torch.linspace(0, np.pi, N).type(torch.float)

##################### forward ########################
#Rf = radon(input, theta=input_angles_degrees, circle=True)
input = torch.tensor(shepp_logan_phantom()).type(torch.float)
img_size = input.shape[0]

reconstruction_angles_degrees =  torch.linspace(0, 180, N).type(torch.float)

radon_class = Radon(img_size, reconstruction_angles_degrees, circle=True)
sinogram = radon_class.forward(input.view(1,1,input.shape[0],input.shape[1]))

radon_uniform = Radon(img_size, input_angles_degrees, circle=True)
sinogram_uniform = radon_uniform.forward(input.view(1,1,input.shape[0],input.shape[1]))

plot_imshow(sinogram[0,0].T, title="Sinogram linspace unsorted")
plot_imshow(sinogram_uniform[0,0].T[torch.argsort(input_angles_degrees).view(N)], title="Sinogram uniform sorted")
plot_imshow(sinogram_uniform[0,0].T, title="Sinogram uniform unsorted")

x_est_GL_t = filterBackprojection2D(sinogram[0,0].T, reconstruction_angles_degrees)
x_est_GL_t_2 = filterBackprojection2D(sinogram_uniform[0,0].T[torch.argsort(input_angles_degrees).view(N)], reconstruction_angles_degrees)
print(x_est_GL_t.shape)
plot_imshow(x_est_GL_t, title="reconstruction")
plot_imshow(x_est_GL_t_2, title="reconstruction uniform")

plt.show()

assert False

# sinogram : 1, 1, img_size, N
noisy_sinogram = add_noise(SNR, sinogram)

org_sino = sinogram.clone()
print(sinogram.shape)
print(sinogram[0,0].shape)

sinogram = sinogram[0,0].transpose(0, 1)
# sinogram :N, img_size,
print(sinogram.shape)
noisy_sinogram = noisy_sinogram[0,0].transpose(0, 1)
print(noisy_sinogram.shape)

plot_imshow(sinogram[torch.argsort(input_angles_degrees).view(N)], title="sinogram sorted")
plot_imshow(noisy_sinogram[torch.argsort(input_angles_degrees.view(N))], title="noisy sinogram sorted")

iradon_class = IRadon(img_size, reconstruction_angles_degrees, out_size=img_size)
#sinogram.view(1,1 ,sinogram.shape[0], sinogram.shape[1])
#out = iradon_class.forward(sinogram[torch.argsort(input_angles_degrees).view(N)].transpose(0,1))
out = iradon_class.forward(org_sino)
print(out.shape)
plot_imshow(out[0,0], title="reconstruction")

plt.show()
assert False
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

plot_2d_scatter(graph_laplacian, titile="GL clean case")
plot_2d_scatter(noisy_graph_laplacian, titile="GL noisy case")

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