import networkx as nx
import numpy as np

def create_adj_mat(index_edge_array, weights=None):
  n = index_edge_array.shape[0]
  neighbors = index_edge_array.shape[1]

  edges = []
  for i in range(0, n):
    for j in range(0, neighbors):
        if weights is not None:
            edges.append((i, index_edge_array[i,j], weights[i,j]))
        else:
            edges.append((i, index_edge_array[i,j]))

  G = nx.Graph()

  if weights is not None:
    G.add_weighted_edges_from(edges)
  else:
    G.add_edges_from(edges)

  return nx.to_numpy_array(G)


def get_degree_matrix(M):
  return np.diag(M.sum(axis=1))

def get_inv_matrix(M):
  return np.linalg.inv(M)

def calc_gaussian_kernel(M, epsilon=1):
  return np.exp(- M / epsilon)


