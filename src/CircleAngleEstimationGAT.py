import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import default_rng
import torch
import torch.nn.functional as F

from utils.Graph import *
from utils.Plotting import *
from utils.Normalization import *



np.set_printoptions(precision=4)

# Parameters
N = 100
K = 4
debug_plot = True
debug_print = False

############### Sampling points on circle ###########################

step = 2*np.pi / N
angle_in_radian = np.linspace(0, 2*np.pi - step,N)

rng = default_rng(1)
# noise = rng.standard_normal(N)
# noisy_angle_in_radian = angle_in_radian + noise

point_on_cricle = np.array([np.cos(angle_in_radian), np.sin(angle_in_radian)]).T
noisy_points_on_circle = point_on_cricle + np.array([rng.normal(0,0.05,N), rng.normal(0,0.05,N)]).T

if debug_plot:
  plt.figure()
  plt.scatter(point_on_cricle[:,0], point_on_cricle[:,1], s=20)
  plt.xticks([-1, -0.5,0,0.5,1])
  plt.yticks([-1, -0.5,0,0.5,1])
  plt.title("clean points")

  plt.figure()
  plt.scatter(noisy_points_on_circle[:,0], noisy_points_on_circle[:,1], s=20)
  plt.xticks([-1, -0.5,0,0.5,1])
  plt.yticks([-1, -0.5,0,0.5,1])
  plt.title("noisy points")

if debug_print:
  print(f"Points on circle: {point_on_cricle}")
  print(f"Angles in radian: {angle_in_radian}")

################# Computing distances #########################

def great_circle_distance(lon1, lat1, lon2, lat2, debug_i):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # if debug_i == 9:
    #   t1 = np.sin(lat1) 
    #   t2 = np.sin(lat2) 
    #   t3 = np.cos(lat1)
    #   t4 = np.cos(lat2)
    #   t5 = np.cos(lon1 - lon2)
    #   arccos_term = t1 * t2 + t3 * t4 * t5
    #   t_arccos = np.arccos(max(min(arccos_term, 1), -1))
    #   if np.isnan(t_arccos):
    #     print(f"t1 {t1}, t2 {t2}, t3 {t3}, t4 {t4}, t5 {t5}")
    #     print(f"lon1 {lon1}, lon2 {lon2}, lat1 {lat1}, lat2 {lat2}")
    #     print(f"arccos_term: {arccos_term} ,result: {t_arccos}")
    #   return t_arccos
    # else:
    return np.arccos( max(min(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2), 1), -1))

distances = np.array([great_circle_distance(point_on_cricle[i,0], point_on_cricle[i,1], point_on_cricle[j,0], point_on_cricle[j,1], i) for i in range(N) for j in range(N)]).reshape((N,N))
distances /= distances.max()

noisy_distances = np.array([great_circle_distance(noisy_points_on_circle[i,0], noisy_points_on_circle[i,1], noisy_points_on_circle[j,0], noisy_points_on_circle[j,1], i) for i in range(N) for j in range(N)]).reshape((N,N))
noisy_distances /= noisy_distances.max()

if debug_print:
  print(f"Distances: {distances}")
  print(f"Distances: {distances.max()}")
  print(f"Distances: {distances.min()}")
  print(f"Distances: {distances.mean()}")


print("Distances done!")

############### Generating Knn graph #########################

graph, classes, edges = generate_knn_from_distances_with_edges(distances, K, ordering='asc', ignoreFirst=True)
noisy_graph, noisy_classes, noisy_edges = generate_knn_from_distances_with_edges(noisy_distances, K, ordering='asc', ignoreFirst=True)

if debug_plot:
  graph_laplacian = calc_graph_laplacian_nx(graph, embedDim=2)
  plot_2d_scatter(graph_laplacian, title=f"Graph Laplacian input graph")

  graph_laplacian_noisy = calc_graph_laplacian_nx(noisy_graph, embedDim=2)
  plot_2d_scatter(graph_laplacian_noisy, title=f"Graph Laplacian noisy input graph")

if debug_print:
  print("Graph: ", graph)
  print("Neighbours", classes)

print("Input Graph done!")

from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
################### GAT class #############################
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_layer_dimension=16, num_classes=2):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_layer_dimension)
        self.conv2 = GATConv(hidden_layer_dimension, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        #x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        #return F.log_softmax(x, dim=1)
        #return x
        return 2 * ((x - torch.min(x)) / (torch.max(x) - torch.min(x)))  - 1


import math
import torch
import torch.linalg as linalg

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
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()


####################### estimate angles ####################
from torch_geometric.data import Data

# thetaSet_t = torch.tensor(thetaSet.copy()).type(torch_type).to(device)
# g_t = torch.tensor(g).type(torch_type).to(device)
# V_t = torch.FloatTensor(vol_org).unsqueeze(0).type(torch_type).to(device)
# ## Generate an experiments
# P_t = torch.zeros((Ntheta,n,n)).type(torch_type).to(device)
# P_t_ = torch.zeros((Ntheta,n,n)).type(torch_type).to(device)

# prep edges:
N_noisy_edges = len(noisy_edges)
edge_index = np.zeros((2, N_noisy_edges))
edge_attribute = np.zeros((N_noisy_edges, 1))
for i in range(N_noisy_edges):
  (n,m) = noisy_edges[i]
  edge_index[0,i] = n
  edge_index[1,i] = m
  edge_attribute[i] = noisy_distances[n,m]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

graph_laplacian_normalized = torch.tensor(normalize_range(graph_laplacian, -1, 1)).type(torch.float).to(device)
graph_laplacian_noisy_normalized =  torch.tensor(normalize_range(graph_laplacian_noisy, -1, 1)).type(torch.float).to(device)

# t_graph_laplacian_noisy = torch.tensor(graph_laplacian_noisy.copy()).type(torch.float).to(device)
# t_noisy_distances = torch.tensor(noisy_distances.copy()).type(torch.float).to(device)
# t_noisy_graph = torch.tensor(noisy_graph.copy()).type(torch.float).to(device)
# t_noisy_edges = torch.tensor(noisy_edges.copy()).type(torch.float).to(device)

t_y = graph_laplacian_normalized
#t_y = torch.tensor(point_on_cricle.copy()).type(torch.float).to(device)
# angles = np.linspace(0, 2 * np.pi, N)
# t_y = torch.tensor(np.array([np.cos(angles), np.sin(angles)]).T).type(torch.float).to(device)
t_x =graph_laplacian_noisy_normalized
#graph_laplacian_noisy_normalized = normalize_range(graph_laplacian_noisy, -1, 1)
#angles = np.arctan2(graph_laplacian_noisy[:,0],graph_laplacian_noisy[:,1]) + np.pi
#print(angles.shape)
#t_x = torch.tensor(angles.reshape((N)).copy()).type(torch.float).to(device)
#print("t x dim:", t_x.dim())


t_edge_index = torch.tensor(edge_index.copy()).type(torch.long).to(device)
t_edge_attribute = torch.tensor(edge_attribute.copy()).type(torch.float).to(device)


data = Data(x=t_x, y=t_y, edge_index=t_edge_index, edge_attribute=t_edge_attribute)
# x (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
# edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
# edge_attr (Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
# y (Tensor, optional) – Graph-level or node-level ground-truth labels with arbitrary shape. (default: None)
# pos (Tensor, optional) – Node position matrix with shape [num_nodes, num_dimensions]. (default: None)
print("number of features:", data.num_node_features)

model = GAT(data.num_node_features, hidden_layer_dimension=2, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    #model.train()
    optimizer.zero_grad()
    out = model(data)
    #out =  2 * ((out - torch.min(out)) / (torch.max(out) - torch.min(out)))  - 1
    
    #loss = F.mse_loss(out, data.y)
    #print(out)

    loss = calculate_2_wasserstein_dist(out, data.y)
    print(f"epoch: {epoch} --- loss: {loss}")
    loss.backward()
    optimizer.step()



model.eval()
pred = model(data)



angles = pred.cpu().detach().numpy()

plot_2d_scatter(angles, title=f"Angle estimation GAT")
# plot_2d_scatter(graph_laplacian_normalized, title=f"Normalized GL")


acc = calculate_2_wasserstein_dist(pred, t_y)
acc_org = calculate_2_wasserstein_dist(graph_laplacian_normalized ,  t_y)
acc_gl = calculate_2_wasserstein_dist(graph_laplacian_noisy_normalized, t_y)

print(f'Accuracy clean GL angles: {acc_org:.4f}')
print(f'Accuracy noisy GL angles: {acc_gl:.4f}')
print(f'Accuracy GAT angles: {acc:.4f}')

# print("Prediction: ", pred)
# print("True angles: ", t_y)


plt.show()