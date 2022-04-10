from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

import time

from utils.Plotting import *
from utils.Graph import *
from utils.obsolete.Data import find_sigma_noise


import torch

from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F



############## PREP #####################

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

# ########### Parameters #############
# data
RESOLUTION = 200
N = 512
DOUBLE_ANGLES = True

SEED = 2022

ADD_CIRCLE_PADDING = True

#graph
K = 9

#noise
SNR=25


debug_plot = True
color_map = plt.cm.get_cmap('gray')
reversed_color_map = color_map.reversed()

# angle_generation = 'linear_spaced'
angle_generation = 'uniform'

use_wandb = False

# ############ Loading input image ###############
input = shepp_logan_phantom() # 400, 400
#input = np.load("toyModel/bunny_2d.npy")
scaleX = RESOLUTION/ input.shape[0]
scaleY = RESOLUTION/ input.shape[1]
input = rescale(input, scale=(scaleX, scaleY), mode='reflect', multichannel=False)


if ADD_CIRCLE_PADDING:
  r = np.ceil(np.sqrt(2 * (RESOLUTION ** 2)) )
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
plot_imshow(sinogram_data[np.argsort(thetas_degree)], "original sinogram")
# Add some noise to this projection.
sinogram_data_noisy = add_noise(SNR, sinogram_data)
plot_imshow(sinogram_data_noisy[np.argsort(thetas_degree)], "noisy sinogram")

##################### Define distances ##########################
# clean case
sinogram_dist = sinogramm_l2_distances(sinogram_data)
graph, classes = generate_knn_from_distances(sinogram_dist, K=K, ordering='asc', ignoreFirst=True)
graph_laplacian = calc_graph_laplacian(graph, embedDim=2)
plot_2d_scatter(graph_laplacian, title=f"Graph Laplacian sinogram graph K = {K}")

# noisy case
sinogram_noisy_dist = sinogramm_l2_distances(sinogram_data_noisy)
graph_noisy, classes_noisy, noisy_edges = generate_knn_from_distances_with_edges(sinogram_noisy_dist, K, ordering='asc', ignoreFirst=True)
graph_laplacian_noisy = calc_graph_laplacian(graph_noisy, embedDim=2)
plot_2d_scatter(graph_laplacian_noisy, title=f"Graph Laplacian noisy sinogram graph")

################### GAT class #############################
class GAT(torch.nn.Module):
    def __init__(self, in_num_features, hidden_layer_dim, out_num_features):
        super().__init__()
        self.conv1 = GATConv(in_num_features, hidden_layer_dim)
        self.conv2 = GATConv(hidden_layer_dim, hidden_layer_dim)
        self.conv3 = GATConv(hidden_layer_dim, out_num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        return x

# class GAT_angles(torch.nn.Module):
#     def __init__(self, in_num_features, hidden_layer_dim, out_num_features):
#         super().__init__()
#         self.conv1 = GATConv(in_num_features, hidden_layer_dim)
#         self.conv2 = GATConv(hidden_layer_dim, hidden_layer_dim)
#         self.conv3 = GATConv(hidden_layer_dim, out_num_features)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         #x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         #return F.log_softmax(x, dim=1)
#         return x

################ GAT #################
# prep edges:
N_noisy_edges = len(noisy_edges)
edge_index = np.zeros((2, N_noisy_edges))
edge_attribute = np.zeros((N_noisy_edges, 1))
for i in range(N_noisy_edges):
  (n,m) = noisy_edges[i]
  edge_index[0,i] = n
  edge_index[1,i] = m
  edge_attribute[i] = sinogram_noisy_dist[n,m]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# t_graph_laplacian_noisy = torch.tensor(graph_laplacian_noisy.copy()).type(torch.float).to(device)
# t_noisy_distances = torch.tensor(noisy_distances.copy()).type(torch.float).to(device)
# t_noisy_graph = torch.tensor(noisy_graph.copy()).type(torch.float).to(device)
# t_noisy_edges = torch.tensor(noisy_edges.copy()).type(torch.float).to(device)

#t_y = torch.tensor(point_on_cricle.copy()).type(torch.float).to(device)
#t_y = torch.tensor(graph_laplacian.copy()).type(torch.float).to(device)

################## denoise sinogram ##########################################
t_y = torch.tensor(sinogram_data.copy()).type(torch.float).to(device)
t_x = torch.tensor(sinogram_data_noisy.copy()).type(torch.float).to(device)
t_edge_index = torch.tensor(edge_index.copy()).type(torch.long).to(device)
t_edge_attribute = torch.tensor(edge_attribute.copy()).type(torch.float).to(device)
sinogram_idx = torch.tensor(np.argsort(thetas_degree)).type(torch.long).to(device)

from torch_geometric.data import Data


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_sinogram = Data(x=t_x, y=t_y, edge_index=t_edge_index, edge_attribute=t_edge_attribute)
model_sinogram = GAT(in_num_features=RESOLUTION, hidden_layer_dim=int(RESOLUTION//2), out_num_features=RESOLUTION).to(device)
optimizer_sinogram = torch.optim.Adam(model_sinogram.parameters(), lr=0.01, weight_decay=5e-4)


################## Estimate angles ###############################################
point_on_cricle = np.array([np.cos(thetas), np.sin(thetas)]).T
# t_y_angles = torch.tensor(thetas.copy()).type(torch.float).to(device)
t_y_angles = torch.tensor(point_on_cricle.copy()).type(torch.float).to(device)
#t_y_angles = torch.tensor(np.linspace(0, 2 * np.pi, N)).type(torch.float).to(device)
t_x_angles = torch.tensor(graph_laplacian_noisy.copy()).type(torch.float).to(device)
#t_x_angles = torch.tensor(sinogram_noisy_dist.copy()).type(torch.float).to(device)

model_angles = GAT(in_num_features=2, hidden_layer_dim=int(N//2), out_num_features=2).to(device)
optimizer_angles = torch.optim.Adam(model_angles.parameters(), lr=0.01, weight_decay=5e-4)
data_angles = Data(x=t_x_angles,y=t_y_angles, edge_index=t_edge_index, edge_attribute=t_edge_attribute)


# train together

model_angles.train()
model_sinogram.train()
for epoch in range(100):
    model_angles.train()
    model_sinogram.train()

    optimizer_angles.zero_grad()
    optimizer_sinogram.zero_grad()
    
    out_sinogram = model_sinogram(data_sinogram)
    out_angles = model_angles(data_angles)

    loss_sinogram = torch.linalg.norm(out_sinogram - data_sinogram.y)
    loss_angles = loss = calculate_2_wasserstein_dist(out_angles, data_angles.y)
    
    print(f"epoch: {epoch} --- loss sino: {loss_sinogram}  --- loss angles: {loss_angles}")
    loss_sinogram.backward()
    loss_angles.backward()
    
    optimizer_angles.step()
    optimizer_sinogram.step()



model_angles.eval()
model_sinogram.eval()

pred_points = model_angles(data_angles)
pred_sinogram = model_sinogram(data_sinogram)

print("Prediction points shape:", pred_points.shape)
pred_angles = torch.atan2(pred_points[:,0],pred_points[:,1]) + torch.pi
print("Prediction angles shape:", pred_angles.shape)

sinogram_idx_estimated = torch.argsort(pred_angles)



plot_2d_scatter(pred_points.cpu().detach().numpy(), title="Estimated angles")


print("sino shape:", pred_sinogram.shape)
print("sino idx shape:", sinogram_idx.shape)
print("sino idx shape:", sinogram_idx_estimated.shape)

plot_imshow(pred_sinogram[sinogram_idx].cpu().detach().numpy(), title='Denoised sinogram', c_map=color_map)
plot_imshow(pred_sinogram[sinogram_idx_estimated].cpu().detach().numpy(), title='Denoised sinogram estimated angles', c_map=color_map)
# correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
# acc = int(correct) / int(data.test_mask.sum())
# print(f'Accuracy: {acc:.4f}')

angles = np.linspace(0, 360, N)
out = iradon(pred_sinogram[sinogram_idx].T.cpu().detach().numpy(), theta=angles,circle=True)
out2 = iradon(pred_sinogram[sinogram_idx_estimated].T.cpu().detach().numpy(), theta=angles,circle=True)

plot_imshow(out, title='Reconstructed gat image', c_map=color_map)
plot_imshow(out2, title='Reconstructed gat image estimated angles', c_map=color_map)
plot_imshow(iradon(sinogram_data[np.argsort(thetas_degree)].T, theta=angles,circle=True), title='Reconstructed original image', c_map=color_map)
plot_imshow(iradon(sinogram_data_noisy[np.argsort(thetas_degree)].T, theta=angles,circle=True), title='Reconstructed noisy image', c_map=color_map)

plt.show()