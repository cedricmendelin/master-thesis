import networkx as nx
import numpy as np
from aspire.basis import ffb_2d
from aspire.classification.align2d import BFRAlign2D
from aspire.classification import RIRClass2D
from aspire.source import ArrayImageSource


"""
Create adjacency matrix from indices and optional weights.
"""
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

"""
Gets the degree matrix of given matrix.
"""
def get_degree_matrix(M):
  return np.diag(M.sum(axis=1))

"""
Gets the inverse of a given matrix.
"""
def get_inv_matrix(M):
  return np.linalg.inv(M)

"""
Calculate gaussian kernel for given matrix.
"""
def calc_gaussian_kernel(M, epsilon=1):
  return np.exp(- M / epsilon)


"""
Calculate L = D - A and retuns first n smallest eigenvectors ignoring the first one.
"""
def calc_graph_laplacian(A, numberOfEvecs=2):
  # Laplacian:
  L = get_degree_matrix(A) - A
  
  #Extract the first n smallest eigenvector of the Laplace matrix.
  eigenValues, eigenVectors = np.linalg.eigh(L)
  idx = np.argsort(eigenValues)

  return eigenVectors[:, idx[1:numberOfEvecs+1]]

"""
Calculate L = D - A and retuns first n smallest eigenvectors ignoring the first one.
Create A from knn indices and optinal weights.
"""
def calc_graph_laplacian_from_knn_indices(indices, weights=None, numberOfEvecs=2):
  A = create_adj_mat(indices, weights)
  return calc_graph_laplacian(A, numberOfEvecs)
  

"""
Compute the rotation invariant distance between two 2D images and define knn graph.
Uses underlying aspire package.
"""
def aspire_knn_with_rotation_invariant_distance(X, K):
  # # Define FB basis to express the data into the FB basis (to avoid approximation error)
  fb = ffb_2d.FFBBasis2D((X.shape[1],X.shape[1]),ell_max=50,dtype=np.float64)
  X_origin = X.copy()
  c1 = fb.evaluate_t(X_origin)
  X_approx = fb.evaluate(c1)

  sr = ArrayImageSource(X_approx)
  rir = RIRClass2D(
      sr,
      fspca_components=400,
      bispectrum_components=np.min([X.shape[0],300]),  # Compressed Features after last PCA stage.
      n_nbor=K,
      n_classes=X_approx.shape[0],
      large_pca_implementation="legacy",
      nn_implementation="legacy",
      bispectrum_implementation="legacy"
      #,    n_angles = M
  )
  # set angles=True to obtain the in-plane angles, but very long to obtain
  # angles=True [not available]
  # Watch out, when angles=True, the correlation is returned istead of the distance
  classes, reflections, rotations, shifts, correlations = rir.classify() 
  return classes, reflections, rotations, shifts, correlations