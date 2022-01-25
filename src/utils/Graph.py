import networkx as nx
import numpy as np
from aspire.basis import ffb_2d
from aspire.classification.align2d import BFRAlign2D
from aspire.classification import RIRClass2D
from aspire.source import ArrayImageSource


"""
Create adjacency matrix from indices and optional weights.
"""
def create_adj_mat(index_edge_array, reflections=None, weights=None):
  n = index_edge_array.shape[0]
  neighbors = index_edge_array.shape[1]
  A = np.zeros((n,n))
  for i in range(n):
    for j in range(neighbors):
        if reflections is None or reflections[i,j]:
          if weights is not None:
              A[i, index_edge_array[i,j]] = weights[i,j]
          else:
              A[i, index_edge_array[i,j]] = 1

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
  # angles=True [not available] => does not compute rotations
  # Watch out, when angles=True, the correlation is returned istead of the distance
  classes, reflections, rotations, shifts, correlations = rir.classify() 
  return classes, reflections, rotations, shifts, correlations


  
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


from aspire.basis import ffb_2d

# Compute Graph Laplacian
def generateNeigbourHoodGraph(distanceMatrix, K,sym=True):
    samples = distanceMatrix.shape[0]
    neighbourMatrix = np.zeros_like(distanceMatrix)

    for i in range(samples):
        distanceRow = distanceMatrix[i, :]
        index = np.argsort(distanceRow)
        neighbourMatrix[i, index[:K+1]] = 1
        if sym:
            neighbourMatrix = (((neighbourMatrix + neighbourMatrix.T)>0)*1).astype('float64')

    return neighbourMatrix

"""
Compute the K-Nearest Neighbour distance matrix.
Computation is based on FFT, thus the number of angles evaluated is equal to the number of 
coefficients in the Fourier-Besel basis.
This is based on the Fast Fourier-Besel decomposition from aspire.

INPUT:
 - x1: shape (batch,N,N), set of input 2D-projections
 - K: number of neighbors
 - ell_max: cf aspire documention, leave it to None if you don't know what it means

 OUTPUT:
  - idx_best_img_knn: indeces of the K-NN best projections
  - angle_est_knn: angles of the K-NN best projections
"""
def rotation_invariant_knn(x1,K=10,ell_max=None):
    fb = ffb_2d.FFBBasis2D((x1.shape[1],x1.shape[1]),ell_max=ell_max,dtype=np.float64)
    c1 = fb.evaluate_t(x1)
    c1=fb.to_complex(c1)
    c1_normalized = c1/np.linalg.norm(c1,axis=1,keepdims=True)


    # Compute the K closest points for each image
    angle_est_knn = np.zeros((x1.shape[0],K))
    idx_best_img_knn = np.zeros((x1.shape[0],K),dtype=int)
    dist_knn = np.zeros((x1.shape[0],K))
    for l in range(x1.shape[0]):
        c_crt = c1_normalized[l][None]
        # Compute correlation on FB basis 
        gamma = np.zeros((c1_normalized.shape[0],len(np.unique(fb.complex_angular_indices ))),dtype=complex)
        for kk in np.unique(fb.complex_angular_indices ):
            gamma[:,kk] = (c1_normalized[:,fb.complex_angular_indices ==kk]*np.conj(c_crt[:,fb.complex_angular_indices ==kk])).sum(1)
        corr = np.abs(np.fft.fft(gamma))
        
        ## Get the best angle for every image
        corr_max_ = np.max(corr,axis=1)
        idx_angle_max_ = np.argmax(corr,axis=1)
        ## Find K images that have the highest correlation
        # avoid sorting the full array
        idx_img = np.argpartition(-corr_max_, K+1,axis=0)[:K+1]
        idx_img = idx_img[idx_img!=l]
        
        idx_best_img = idx_img[np.argsort(corr_max_[idx_img])[::-1]] # indices of the best match of images
        idx_angle_max = idx_angle_max_[idx_best_img] # indices of the angles corresponding to bets images
        corr_max = corr_max_[idx_best_img] # value of the correlation of the best images

        angle_est = 2*np.pi*idx_angle_max/corr.shape[1] # angle corresponding the the best matches in radian

        idx_best_img_knn[l] = idx_best_img
        angle_est_knn[l] = angle_est
        dist_knn[l] = 1-corr_max

    return dist_knn, idx_best_img_knn, angle_est_knn



"""
Compute the K-Nearest Neighbour distance matrix. 
Warning, unless the previous function, this function store the full distance matrix.
This will work only if enough RAM is available.
This function rests on the same implementation as the previous one, but avoid the for loop over all the projections.

Computation is based on FFT, thus the number of angles evaluated is equal to the number of 
coefficients in the Fourier-Besel basis.
This is based on the Fast Fourier-Besel decomposition from aspire.

INPUT:
 - x1: shape (batch,N,N), set of input 2D-projections
 - K: number of neighbors
 - ell_max: cf aspire documention, leave it to None if you don't know what it means

 OUTPUT:
  - idx_best_img_knn: indeces of the K-NN best projections
  - angle_est_knn: angles of the K-NN best projections
"""
def rotation_invariant_knn_full(x1,K=10,ell_max=None):
    fb = ffb_2d.FFBBasis2D((x1.shape[1],x1.shape[1]),ell_max=ell_max,dtype=np.float64)
    c1 = fb.evaluate_t(x1)
    c1=fb.to_complex(c1)
    c1_normalized = c1/np.linalg.norm(c1,axis=1,keepdims=True)

    gamma = np.zeros((c1_normalized.shape[0],c1_normalized.shape[0],len(np.unique(fb.complex_angular_indices ))),dtype=complex)
    for kk in np.unique(fb.complex_angular_indices ):
        tmp = c1_normalized[:,fb.complex_angular_indices ==kk]
        gamma[:,:,kk] = (np.conj(tmp[:,None,:])*tmp[None,:,:]).sum(2)
    corr = np.abs(np.fft.fft(gamma))

    ## Get the best angle for every image
    corr_max_ = np.max(corr,axis=-1)
    idx_angle_max_ = np.argmax(corr,axis=-1)
    
    from sklearn.neighbors import NearestNeighbors
    knn_distance_based = NearestNeighbors(n_neighbors=K, metric="precomputed").fit(1-corr_max_)
    dist_knn, idx_best_img_knn = knn_distance_based.kneighbors(return_distance=True)
    angle_est = 2*np.pi*idx_angle_max_/corr.shape[-1]
    angle_est = np.array([angle_est[np.arange(idx_best_img_knn.shape[0]),idx_best_img_knn[:,k]] for k in range(K)]).T

    return dist_knn, idx_best_img_knn, angle_est