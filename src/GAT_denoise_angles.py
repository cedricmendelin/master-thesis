
from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon

from utils.Graph import *
from utils.Plotting import *
from utils.Normalization import normalize_range
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

def calculate_2_wasserstein_dist(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''

    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = torch.linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()

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

SEED = 2022

#graph
#K = 8 # good one
K = 8

#noise
SNR=25


torch.manual_seed(2022)

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

x = torch.cos(reconstruction_angles)
y = torch.sin(reconstruction_angles)

print(x.shape)
print(y.shape)

reconstruction_points = torch.stack((x,y)).T

############### Forward #########################

radon_class = Radon(RESOLUTION, input_angles_degrees, circle=True)

sinogram = radon_class.forward(input_t.view(1, 1, RESOLUTION, RESOLUTION))[0,0]
noisy_sinogram = add_noise(SNR, sinogram)

print("Loss sinogramm", torch.linalg.norm(sinogram - noisy_sinogram))
print("Loss sinogramm.T", torch.linalg.norm(sinogram.T - noisy_sinogram.T))


sorted, idx = torch.sort(input_angles_degrees, 0)
idx = idx.view(N)

plot_imshow(sinogram.T[idx], title="Sinogram uniform")
plot_imshow(noisy_sinogram.T[idx], title="Sinogram noisy")

############################## Distances ###########################
# distances:
distances = distance_matrix(sinogram.T, sinogram.T)
distances /= distances.max()

noisy_distances = distance_matrix(noisy_sinogram.T, noisy_sinogram.T)
noisy_distances /= noisy_distances.max()

# k-nn:

# for k in range(5, 12):
#K = k
graph, classes, edges = generate_knn_from_distances_with_edges(distances, K, ordering='asc', ignoreFirst=True)

noisy_graph, noisy_classes, noisy_edges = generate_knn_from_distances_with_edges(noisy_distances, K, ordering='asc', ignoreFirst=True)

# gl:
graph_laplacian = calc_graph_laplacian(graph, embedDim=2)
noisy_graph_laplacian = calc_graph_laplacian(noisy_graph, embedDim=2)

plot_2d_scatter(graph_laplacian, title=f"GL clean case K={K}")
plot_2d_scatter(noisy_graph_laplacian, title="GL noisy case")
plot_2d_scatter(reconstruction_points.numpy(), title='reconstruction points')

print(graph_laplacian.max())
print(graph_laplacian.min())

_, gl_idx, _ = estimate_angles(graph_laplacian)
_, noisy_gl_idx, _ = estimate_angles(noisy_graph_laplacian)

t_gl_idx = torch.from_numpy(gl_idx).type(torch.long)
t_noisy_gl_idx = torch.from_numpy(noisy_gl_idx).type(torch.long)

x_est_GL_t = filterBackprojection2D(sinogram.T[t_gl_idx], reconstruction_angles_degrees)
plot_imshow(x_est_GL_t.cpu().detach().numpy(), title="Reconstruction clean with gl angles")

x_est_GL_t = filterBackprojection2D(noisy_sinogram.T[t_noisy_gl_idx], reconstruction_angles_degrees)
plot_imshow(x_est_GL_t.cpu().detach().numpy(), title="Reconstruction noisy with gl angles")


################### GAT class #############################
class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim, heads = 1, dropout=0.5):
        super().__init__()
        
        #assert num_layers > 0
        # in_dim = hidden_dim * heads
        self.convs =  torch.nn.ModuleList()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.dropout = dropout

        # layer 1:
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(in_dim, hidden_dim, heads))

        # last layer:
        self.convs.append(GATConv(hidden_dim * heads, out_dim, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if len(self.convs) - 1 != layer:
                x = F.elu(x)
                x = F.dropout(x, self.dropout)

        return x

class GAT2(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim=2, heads = 1, dropout=0.5):
        super().__init__()
        
        #assert num_layers > 0
        # in_dim = hidden_dim * heads
        self.convs =  torch.nn.ModuleList()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.dropout = dropout

        # layer 1:
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(in_dim, hidden_dim, heads))

        # last layer:
        self.convs.append(GATConv(hidden_dim * heads, out_dim, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if len(self.convs) - 1 != layer:
                x = F.elu(x)
                x = F.dropout(x, self.dropout)

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
t_y = reconstruction_points.clone().to(device)
#t_y = reconstruction_angles.clone().view(N,1).to(device)
print("t_y shape", t_y.shape)
#norm_laplacian = normalize_range(noisy_graph_laplacian, -1, 1)
#norm_laplacian = noisy_graph_laplacian * (1/ noisy_graph_laplacian.max())
#t_x = torch.tensor(norm_laplacian).type(torch.float).to(device)
#laplacian_angles, _, laplacian_angles_sorted = estimate_angles(noisy_graph_laplacian)
#t_x = torch.tensor(laplacian_angles).type(torch.float).view(N,1).to(device)
t_x = torch.tensor(noisy_distances).type(torch.float).to(device)
print("t_x shape", t_x.shape)


# plot_2d_scatter(pred_angles.cpu().detach().numpy(), title=f"Angles denoised")
# plot_2d_scatter(np.array([np.cos(laplacian_angles), np.sin(laplacian_angles)]).T, title=f" Laplacian Angles")
# plot_2d_scatter(np.array([np.cos(laplacian_angles_sorted), np.sin(laplacian_angles_sorted)]).T, title=f" Laplacian Angles sorted")



#t_x = torch.tensor()
t_edge_index = torch.tensor(edge_index.copy()).type(torch.long).to(device)
t_edge_attribute = torch.tensor(edge_attribute.copy()).type(torch.float).to(device)

from torch_geometric.data import Data



# print("Loss angles non-normalized:", calculate_2_wasserstein_dist(torch.tensor(graph_laplacian).type(torch.float).to(device), t_y))
# print("Loss angles:", calculate_2_wasserstein_dist(t_x, t_y)) #angles: 0.0240

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data_angles = Data(x=t_x, y=t_y, edge_index=t_edge_index, edge_attr=t_edge_attribute)
data_angles = Data(x=t_x, y=t_y, edge_index=t_edge_index)
#h = 1
# GATdef __init__(self, in_dim, hidden_dim, num_layers, out_dim, heads = 1, dropout=0.5):
# def __init__(self, in_dim, hidden_dim, num_layers, out_dim, heads = 1, dropout=0.5):
# model_angles = GAT2(in_dim=data_angles.num_node_features, hidden_dim=2, num_layers=2, out_dim=2, dropout=0.03).to(device)
model_angles = GAT2(
    in_dim=data_angles.num_node_features, 
    hidden_dim=data_angles.num_node_features, 
    num_layers=3,
    out_dim=2, dropout=0.03).to(device)
#heads = 4
#model_angles = GAT2(in_dim=N, hidden_dim=N // heads,num_layers=2, out_dim=2, heads=heads, dropout=0.05).to(device)
optimizer_angles = torch.optim.Adam(model_angles.parameters(), lr=0.01, weight_decay=5e-4)

# GET with 1d angle:
# AssertionError: Static graphs not supported in 'GATConv'

ab = torch.ones(N) / N
ab.to(device)

from ot import emd2, dist
model_angles.train()
for epoch in range(50):
    #model_angles.train()
    
    optimizer_angles.zero_grad()
    out_angles = model_angles(data_angles)
    # loss_angles = torch.linalg.norm(out_angles - data_angles.y)
    loss_angles = calculate_2_wasserstein_dist(out_angles, data_angles.y)
    # M = dist(out_angles, data_angles.y)#.to(device)
    # loss_angles2 = emd2(ab.clone().to(device), ab.clone().to(device), M, numItermax=120000)
    # loss_angles.to(device)
    #loss_angles = torch.linalg.norm(out_angles - data_angles.y)
    if epoch % 5 == 0:
        plot_2d_scatter(out_angles.cpu().detach().numpy(), title=f"Angles denoised - {epoch}")
    # print(loss_angles)
    #loss_angles = ot.wasserstein_1d(out_angles, data_angles.y)
    #loss_angles = scipy.stats.wasserstein_distance(out_angles, data_angles.y)
    
    print(f"epoch: {epoch} --- loss angles: {loss_angles}  ") # --- loss angles2: {loss_angles2}
    loss_angles.backward()
    
    optimizer_angles.step()

model_angles.eval()
pred_angles = model_angles(data_angles)

# x = torch.cos(reconstruction_angles)
# y = torch.sin(reconstruction_angles)

# print(x.shape)
# print(y.shape)

# reconstruction_points = torch.stack((x,y)).T
#pred_points = np.array([np.cos(pred_angles.cpu().detach().numpy()), np.sin(pred_angles.cpu().detach().numpy())]).T

plot_2d_scatter(pred_angles.cpu().detach().numpy(), title=f"Angles denoised")
#plot_2d_scatter(pred_points, title=f"Angles denoised")

# x_est_GL_t = filterBackprojection2D(pred_angles[idx], reconstruction_angles_degrees)
# plot_imshow(x_est_GL_t.cpu().detach().numpy(), title="reconstruction denoised sorted ")

# x_est_GL_t = filterBackprojection2D(pred_angles, reconstruction_angles_degrees)
# plot_imshow(x_est_GL_t.cpu().detach().numpy(), title="reconstruction denoised unsorted")

plt.show()

