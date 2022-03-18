
from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon

from utils.Graph import *
from utils.Plotting import *
from utils.pytorch_radon.radon import *
import matplotlib.pyplot as plt

from skimage.transform import  rescale, radon, iradon

import torch
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from scipy.spatial import distance_matrix
import torch.nn.functional as F
import math
import imageio

import numpy as np
from numpy.random import default_rng

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##################### Helpers ################
def add_noise(SNR, sinogram):
    VARIANCE = 10 ** (-SNR/10) * (torch.std(sinogram) **2) 
    noise = torch.randn(sinogram.shape[0], sinogram.shape[1]) * torch.sqrt(VARIANCE)
    return sinogram + noise


################## Parameters ########################
RESOLUTION = 200
N = 512

SEED = 2022

#graph
K = 9

#noise
SNR=25

#################### Input #####################
#input = imageio.imread('src/maps/toy_image.png')
input = shepp_logan_phantom()

scaleX = RESOLUTION/ input.shape[0]
scaleY = RESOLUTION/ input.shape[1]
input = rescale(input, scale=(scaleX, scaleY), mode='reflect', multichannel=False)

input_t = torch.from_numpy(input).type(torch.float)#.to(device)

plot_imshow(input, title="Input image")

################# Angles ##########################
from torch.distributions.uniform import Uniform
normal = Uniform(torch.tensor([0.0]), 2 * torch.pi)
input_angles = normal.sample((N,))
input_angles = torch.cat([input_angles, torch.remainder(input_angles + torch.pi, 2 * torch.pi)])
N = N * 2

input_angles_degrees = torch.rad2deg(input_angles).type(torch.float)

reconstruction_angles_degrees =  torch.linspace(0, 360, N).type(torch.float)
reconstruction_angles =  torch.linspace(0, 2 * torch.pi, N).type(torch.float)

############### Forward #########################

radon_class = Radon(RESOLUTION, input_angles_degrees, circle=True)

sinogram = radon_class.forward(input_t.view(1, 1, RESOLUTION, RESOLUTION))[0,0]
noisy_sinogram = add_noise(SNR, sinogram)

print("Loss sinogramm", torch.linalg.norm(sinogram - noisy_sinogram))
print("Loss sinogramm", torch.linalg.norm(sinogram.T - noisy_sinogram.T))


sorted, idx = torch.sort(input_angles_degrees, 0)
idx = idx.view(N)

plot_imshow(sinogram.T[idx], title="Sinogram uniform")
plot_imshow(noisy_sinogram.T[idx], title="Sinogram noisy")

x_est_GL_t = filterBackprojection2D(sinogram.T[idx], reconstruction_angles_degrees)
plot_imshow(x_est_GL_t, title="reconstruction")

x_est_GL_t_noisy = filterBackprojection2D(noisy_sinogram.T[idx], reconstruction_angles_degrees)
plot_imshow(x_est_GL_t_noisy, title="reconstruction noisy sinogram")

############################## Distances ###########################
# distances:
distances = distance_matrix(sinogram.T, sinogram.T)
distances /= distances.max()

noisy_distances = distance_matrix(noisy_sinogram.T, noisy_sinogram.T)
noisy_distances /= noisy_distances.max()

# k-nn:

# for k in range(5, 12):
#     K = k
graph, classes, edges = generate_knn_from_distances_with_edges(distances, K, ordering='asc', ignoreFirst=True)

noisy_graph, noisy_classes, noisy_edges = generate_knn_from_distances_with_edges(noisy_distances, K, ordering='asc', ignoreFirst=True)

# gl:
graph_laplacian = calc_graph_laplacian(graph, embedDim=2)
noisy_graph_laplacian = calc_graph_laplacian(noisy_graph, embedDim=2)

plot_2d_scatter(graph_laplacian, title=f"GL clean case K={K}")
plot_2d_scatter(noisy_graph_laplacian, title="GL noisy case")

################### GAT class #############################
# class GAT(torch.nn.Module):
#     def __init__(self, num_features, num_layers, dropout = 0.5):
#         super().__init__()
        
#         #assert num_layers > 0
        
#         self.convs = []
#         self.num_layers = num_layers
#         self.dropout = dropout

#         for i in range(num_layers):
#             self.convs.append(GATConv(num_features, num_features))

#     def forward(self, x, edge_index):
#         for layer, conv in enumerate(self.convs):
#             x = conv(x, edge_index)
#             if len(self.convs) - 1 != layer:
#                 x = F.relu(x)
#                 x = F.dropout(x, self.dropout)

#         return x

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_layers = 3, dropout = 0.5):
        super().__init__()
        self.conv1 = GATConv(num_features, num_features)
        #self.conv2 = GATConv(num_features, num_features)
        self.conv3 = GATConv(num_features, num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)

        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        return x

################ GAT #################
# prep edges:
N_noisy_edges = len(noisy_edges)
edge_index = np.zeros((2, N_noisy_edges))
edge_attribute = np.zeros((N_noisy_edges, 1))
noisy_edges_list = list(noisy_edges)
for i in range(N_noisy_edges):
  (n,m) = noisy_edges_list[i]
  edge_index[0,i] = n
  edge_index[1,i] = m
  edge_attribute[i] = noisy_distances[n,m]

# GAT setup:
t_y = sinogram.T.clone().to(device)
print("t_y shape", t_y.shape)
t_x = noisy_sinogram.T.clone().to(device)
t_edge_index = torch.tensor(edge_index.copy()).type(torch.long).to(device)
t_edge_attribute = torch.tensor(edge_attribute.copy()).type(torch.float).to(device)

from torch_geometric.data import Data


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_sinogram = Data(x=t_x, y=t_y, edge_index=t_edge_index, edge_attr=t_edge_attribute)
model_sinogram = GAT(num_features=RESOLUTION, num_layers=2).to(device)
optimizer_sinogram = torch.optim.Adam(model_sinogram.parameters(), lr=0.01, weight_decay=5e-4)

model_sinogram.train()
for epoch in range(250):
    #model_sinogram.train()

    optimizer_sinogram.zero_grad()
    out_sinogram = model_sinogram(data_sinogram)
    loss_sinogram = torch.linalg.norm(out_sinogram - data_sinogram.y)
    
    print(f"epoch: {epoch} --- loss sino: {loss_sinogram} ")
    loss_sinogram.backward()
    
    optimizer_sinogram.step()

model_sinogram.eval()
pred_sinogram = model_sinogram(data_sinogram)
plot_imshow(pred_sinogram[idx].cpu().detach().numpy(), title='Denoised sinogram sorted', c_map=color_map)
plot_imshow(pred_sinogram.cpu().detach().numpy(), title='Denoised sinogram unsorted', c_map=color_map)

x_est_GL_t = filterBackprojection2D(pred_sinogram[idx], reconstruction_angles_degrees)
plot_imshow(x_est_GL_t.cpu().detach().numpy(), title="reconstruction denoised sorted ")

x_est_GL_t = filterBackprojection2D(pred_sinogram, reconstruction_angles_degrees)
plot_imshow(x_est_GL_t.cpu().detach().numpy(), title="reconstruction denoised unsorted")

plt.show()

assert False