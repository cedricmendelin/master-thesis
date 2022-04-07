import networkx as nx
import numpy as np
import scipy

# from aspire.basis import ffb_2d
# from aspire.classification.align2d import BFRAlign2D
# from aspire.classification import RIRClass2D
# from aspire.source import ArrayImageSource
from sklearn import neighbors

from .GraphAlign import GAlign

from scipy import sparse
from scipy.sparse.linalg import eigsh

"""
Create adjacency matrix from indices and optional weights.
"""
def create_adj_mat(index_edge_array, weights=None, sym=True):
  n = index_edge_array.shape[0]
  neighbors = index_edge_array.shape[1]
  A = np.zeros((n,n))
  for i in range(n):
    for j in range(neighbors):
      if weights is not None:
        A[i, index_edge_array[i,j]] = weights[i,j]
        if sym:
          A[index_edge_array[i,j], i] = weights[i,j]
      else:
        A[i, index_edge_array[i,j]] = 1
        if sym:
          A[index_edge_array[i,j], i] = 1

  return A


def create_adj_mat_nx(index_edge_array, reflections=None, weights=None):
  n = index_edge_array.shape[0]
  neighbors = index_edge_array.shape[1]
  edges = []
  for i in range(0, n):
    for j in range(0, neighbors):
        if reflections is None or reflections[i,j]:
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

def get_degree_matrix_inv(M):
  return np.diag(1 / M.sum(axis=1))

def get_degree_matrix_inv_half(M):
  return np.diag(np.sqrt(1 / M.sum(axis=1)))

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


def extract_embedding(L, embedDim):
  L=sparse.csr_matrix(L)
  eigenValues, eigenVectors = eigsh(L,k=embedDim + 1,which='SM')
  return eigenVectors[:, 1:embedDim + 1 ]

"""
Calculate L = D - A and retuns first n smallest eigenvectors ignoring the first one.
"""
def calc_graph_laplacian(A, embedDim=2):
  # Laplacian:
  L = get_degree_matrix(A) - A
  return extract_embedding(L, embedDim)

def calc_graph_laplacian_symmetric(A, embedDim=2):
  # Laplacian:
  D_inv_half = get_degree_matrix_inv_half(A)
  I = np.identity(A.shape[0])
  L = I - D_inv_half @ A @ D_inv_half
  
  return extract_embedding(L, embedDim)
  
def calc_graph_laplacian_rw(A, embedDim=2):
  # Laplacian:
  D_inv = get_degree_matrix_inv(A)
  I = np.identity(A.shape[0])
  L = I - D_inv @ A
  
  return extract_embedding(L, embedDim)

def calc_graph_laplacian_density_invariant(A, embedDim=2):
  # Laplacian:
  D_inv = get_degree_matrix_inv(A)
  W =  D_inv @ A @ D_inv
  DW_inv = np.diag(1 / W.sum(axis=1))
  L = DW_inv @ W - np.identity(A.shape[0])
  
  return extract_embedding(L, embedDim)

def calc_graph_laplacian_nx(W, embedDim = 2):
    G = nx.convert_matrix.from_numpy_matrix(W)
    if not nx.is_connected(G):
        print("Graph is not connected!!")
    L = nx.laplacian_matrix(G)
    eVals, eVecs = scipy.sparse.linalg.eigsh(L,k=embedDim + 1,which='SM')
    return eVecs[:,1:embedDim+1]


"""
Calculate L = D - A and retuns first n smallest eigenvectors ignoring the first one.
Create A from knn indices and optinal weights.
"""
def calc_graph_laplacian_from_knn_indices(indices, weights=None, embedDim=2):
  A = create_adj_mat(indices, weights)
  return calc_graph_laplacian(A, embedDim)
  

def align_3d_embedding_to_shpere(embedding, debug=False):
  n = embedding.shape[0]
  fibo_sphere = fibonacci_sphere(samples=n)
  G_aligned = GAlign("embedding", emb1=embedding, emb2=fibo_sphere).get_align()
  result = fibo_sphere[G_aligned]

  if debug:
    for i in range(n):
      if i % 100 == 0:
        print(f"Actual {embedding[i]}, estimated {result[i]}")
  
    dist = np.linalg.norm(embedding - result)
    print(dist/n)

  return result


# """
# Compute the rotation invariant distance between two 2D images and define knn graph.
# Uses underlying aspire package.
# """
# def aspire_knn_with_rotation_invariant_distance(X, K):
#   # # Define FB basis to express the data into the FB basis (to avoid approximation error)
#   fb = ffb_2d.FFBBasis2D((X.shape[1],X.shape[1]),ell_max=50,dtype=np.float64)
#   X_origin = X.copy()
#   c1 = fb.evaluate_t(X_origin)
#   X_approx = fb.evaluate(c1)

#   sr = ArrayImageSource(X_approx)
#   rir = RIRClass2D(
#       sr,
#       fspca_components=400,
#       bispectrum_components=np.min([X.shape[0],300]),  # Compressed Features after last PCA stage.
#       n_nbor=K,
#       n_classes=X_approx.shape[0],
#       large_pca_implementation="legacy",
#       nn_implementation="legacy",
#       bispectrum_implementation="legacy"
#       #,    n_angles = M
#   )
#   # set angles=True to obtain the in-plane angles, but very long to obtain
#   # angles=True [not available] => does not compute rotations
#   # Watch out, when angles=True, the correlation is returned istead of the distance
#   classes, reflections, rotations, shifts, correlations = rir.classify() 
#   return classes, reflections, rotations, shifts, correlations


def sampling_sphere(Ntheta):
    th = np.random.random(Ntheta) * np.pi * 2
    x = np.random.random(Ntheta) * 2 - 1
    out = np.array([np.cos(th) * np.sqrt(1 - x**2), np.sin(th) * np.sqrt(1 - x**2),x]).T
    return out
  
def fibonacci_sphere(samples=1000):
    points = np.zeros((samples, 3))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i] = np.array([x,y,z])

    return points


# from aspire.basis import ffb_2d

def generate_knn_from_distances(distanceMatrix, K, sym=True, ordering='desc', dataType='float64', ignoreFirst=False):
    samples = distanceMatrix.shape[0]
    A = np.zeros_like(distanceMatrix)
    neighbors = np.zeros((samples, K))

    for i in range(samples):
        distanceRow = distanceMatrix[i, :]

        if ordering == 'desc':
          if ignoreFirst:
            index = np.argsort(-distanceRow)[1:K+1]
          else:
            index = np.argsort(-distanceRow)[0:K]
        else:
          if ignoreFirst:
            index = np.argsort(distanceRow)[1:K+1]
          else:
            index = np.argsort(distanceRow)[0:K]

        A [i][index] = 1
        neighbors[i] = index

        if sym:
            A  = (((A  + A .T) > 0) * 1 ).astype(dataType)

    return A , neighbors.astype(np.int)

def generate_knn_from_distances_with_edges(distanceMatrix, K, sym=True, ordering='desc', dataType='float64', ignoreFirst=False):
    samples = distanceMatrix.shape[0]
    A = np.zeros_like(distanceMatrix)
    neighbors = np.zeros((samples, K))
    edges = set()

    for i in range(samples):
        distanceRow = distanceMatrix[i, :]

        if ordering == 'desc':
          if ignoreFirst:
            index = np.argsort(-distanceRow)[1:K+1]
          else:
            index = np.argsort(-distanceRow)[0:K]
        else:
          if ignoreFirst:
            index = np.argsort(distanceRow)[1:K+1]
          else:
            index = np.argsort(distanceRow)[0:K]

        A [i][index] = 1
        neighbors[i] = index

        for j in index:
          edges.add((i,j))
          edges.add((j,i))

        if sym:
            A  = (((A  + A .T) > 0) * 1 ).astype(dataType)
            

    return A , neighbors.astype(np.int), edges
    
import torch
def generate_knn_from_distances_with_edges_t(distanceMatrix, K, sym=True, ordering='desc', dataType='float64', ignoreFirst=False):
    samples = distanceMatrix.shape[0]
    A = torch.zeros_like(distanceMatrix)
    neighbors = torch.zeros((samples, K))
    edges = []

    for i in range(samples):
        distanceRow = distanceMatrix[i, :]

        if ordering == 'desc':
          if ignoreFirst:
            index = torch.argsort(-distanceRow, 0)[1:K+1]
          else:
            index = torch.argsort(-distanceRow, 0)[0:K]
        else:
          if ignoreFirst:
            index = torch.argsort(distanceRow, 0)[1:K+1]
          else:
            index =torch.argsort(-distanceRow, 0)[0:K]

        A [i][index] = 1
        neighbors[i] = index

        for j in index:
          edges.append((i,j))

        if sym:
            A  = (((A  + A .T) > 0) * 1 ).astype(dataType)

    return A , neighbors.type(torch.int), edges


"""
Compute the K-Nearest Neighbour distance matrix.
Computation is based on FFT, thus the number of angles evaluated is equal to the number of
coefficients in the Fourier-Besel basis.
This is based on the Fast Fourier-Besel decomposition from aspire.
INPUT:
 - x1: shape (batch,N,N), set of input 2D-projections
 - K: number of neighbors
 - ell_max: cf aspire documention, leave it to None if you don't know what it means
 - refl_bool: either or not compute reflection invariant distance
 OUTPUT:
  - dist_knn: 1-higest correlation
  - idx_best_img_knn: indeces of the K-NN best projections
  - angle_est_knn: angles of the K-NN best projections
  - refl_knn: boolean array indicating is the closest image is with or without reflection
"""
# def rotation_invariant_knn(x1,K=10,ell_max=None,refl_bool=True):
#     fb = ffb_2d.FFBBasis2D((x1.shape[1],x1.shape[1]),ell_max=ell_max,dtype=np.float64)
#     c1 = fb.evaluate_t(x1)
#     c1=fb.to_complex(c1)
#     c1_normalized = c1/np.linalg.norm(c1,axis=1,keepdims=True)
#     # Compute the K closest points for each image
#     angle_est_knn = np.zeros((x1.shape[0],K))
#     idx_best_img_knn = np.zeros((x1.shape[0],K),dtype=int)
#     dist_knn = np.zeros((x1.shape[0],K))
#     refl_knn = np.zeros((x1.shape[0],K),dtype=bool)
#     for l in range(x1.shape[0]):
#         c_crt = c1_normalized[l][None]
#         # Compute correlation on FB basis
#         gamma1 = np.zeros((c1_normalized.shape[0],len(np.unique(fb.complex_angular_indices ))),dtype=complex)
#         gamma2 = np.zeros((c1_normalized.shape[0],len(np.unique(fb.complex_angular_indices ))),dtype=complex) # reflection
#         for kk in np.unique(fb.complex_angular_indices):
#             gamma1[:,kk] = (c1_normalized[:,fb.complex_angular_indices ==kk]*np.conj(c_crt[:,fb.complex_angular_indices ==kk])).sum(1)
#             gamma2[:,kk] = (c1_normalized[:,fb.complex_angular_indices ==kk]*(c_crt[:,fb.complex_angular_indices ==kk])).sum(1)
#             # gamma2[:,kk] = (c1_normalized[:,fb.complex_angular_indices == kk]*np.conj(c_crt_refl[:,fb.complex_angular_indices ==kk])).sum(1)
#         corr1 = np.abs(np.fft.fft(gamma1))
#         corr2 = np.abs(np.fft.fft(gamma2))
#         if refl_bool:
#             corr = np.maximum(corr1, corr2)
#         else:
#             corr = corr1
#         refl = corr!=corr1
#         ## Get the best angle for every image
#         corr_max_ = np.max(corr,axis=1)
#         idx_angle_max_ = np.argmax(corr,axis=1)
#         ## Find K images that have the highest correlation
#         # avoid sorting the full array
#         idx_img = np.argpartition(-corr_max_, K,axis=0)[:K]
#         # idx_img = idx_img[idx_img!=l]
#         idx_best_img = idx_img[np.argsort(corr_max_[idx_img])[::-1]] # indices of the best match of images
#         idx_angle_max = idx_angle_max_[idx_best_img] # indices of the angles corresponding to bets images
#         corr_max = corr_max_[idx_best_img] # value of the correlation of the best images
#         angle_est = 2*np.pi*idx_angle_max/corr.shape[1] # angle corresponding the the best matches in radian
#         idx_best_img_knn[l] = idx_best_img
#         angle_est_knn[l] = angle_est
#         dist_knn[l] = 1-corr_max
#         refl_knn[l] = refl[idx_best_img,idx_angle_max]
#     return dist_knn, idx_best_img_knn, angle_est_knn, refl_knn