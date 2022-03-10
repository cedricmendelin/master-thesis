import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import default_rng
import torch
import torch.nn.functional as F

from utils.Graph import *
from utils.Plotting import *



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

from torch_geometric.nn import GCNConv, GATConv
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

        return F.log_softmax(x, dim=1)


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

# t_graph_laplacian_noisy = torch.tensor(graph_laplacian_noisy.copy()).type(torch.float).to(device)
# t_noisy_distances = torch.tensor(noisy_distances.copy()).type(torch.float).to(device)
# t_noisy_graph = torch.tensor(noisy_graph.copy()).type(torch.float).to(device)
# t_noisy_edges = torch.tensor(noisy_edges.copy()).type(torch.float).to(device)

t_y = torch.tensor(point_on_cricle.copy()).type(torch.float).to(device)
t_x = torch.tensor(noisy_distances.copy()).type(torch.float).to(device)
t_edge_index = torch.tensor(edge_index.copy()).type(torch.long).to(device)
t_edge_attribute = torch.tensor(edge_attribute.copy()).type(torch.float).to(device)



data = Data(x=t_x,y=t_y, edge_index=t_edge_index, edge_attribute=align_3d_embedding_to_shpere)
# x (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
# edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
# edge_attr (Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
# y (Tensor, optional) – Graph-level or node-level ground-truth labels with arbitrary shape. (default: None)
# pos (Tensor, optional) – Node position matrix with shape [num_nodes, num_dimensions]. (default: None)
print("number of features:", data.num_node_features)

model = GAT(N,hidden_layer_dimension=int(N/2)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.cosine_embedding_loss(out, data.y)
    print(f"epoch: {epoch} --- loss: {loss}")
    loss.backward()
    optimizer.step()



model.eval()
pred = model(data)
print(pred)
acc = torch.linalg.norm(pred - t_y)

print(f'Accuracy: {acc:.4f}')

angles = pred.cpu().detach().numpy()

plot_2d_scatter(angles, title=f"Angle estimation GAT")

plt.show()
