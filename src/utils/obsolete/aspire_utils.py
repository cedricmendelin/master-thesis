import sys
import numpy as np
sys.path.insert(0, '..')
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
 - refl_bool: either or not compute reflection invariant distance

 OUTPUT:
  - dist_knn: 1-higest correlation
  - idx_best_img_knn: indeces of the K-NN best projections
  - angle_est_knn: angles of the K-NN best projections
  - refl_knn: boolean array indicating is the closest image is with or without reflection
"""
def rotation_invariant_knn(x1,K=10,ell_max=None,refl_bool=True):
    fb = ffb_2d.FFBBasis2D((x1.shape[1],x1.shape[1]),ell_max=ell_max,dtype=np.float64)
    c1 = fb.evaluate_t(x1)
    c1=fb.to_complex(c1)
    c1_normalized = c1/np.linalg.norm(c1,axis=1,keepdims=True)

    # Compute the K closest points for each image
    angle_est_knn = np.zeros((x1.shape[0],K))
    idx_best_img_knn = np.zeros((x1.shape[0],K),dtype=int)
    dist_knn = np.zeros((x1.shape[0],K))
    refl_knn = np.zeros((x1.shape[0],K),dtype=bool)
    for l in range(x1.shape[0]):
        c_crt = c1_normalized[l][None]
        # Compute correlation on FB basis 
        gamma1 = np.zeros((c1_normalized.shape[0],len(np.unique(fb.complex_angular_indices ))),dtype=complex)
        gamma2 = np.zeros((c1_normalized.shape[0],len(np.unique(fb.complex_angular_indices ))),dtype=complex) # reflection
        for kk in np.unique(fb.complex_angular_indices):
            gamma1[:,kk] = (c1_normalized[:,fb.complex_angular_indices ==kk]*np.conj(c_crt[:,fb.complex_angular_indices ==kk])).sum(1)
            gamma2[:,kk] = (c1_normalized[:,fb.complex_angular_indices ==kk]*(c_crt[:,fb.complex_angular_indices ==kk])).sum(1)
        corr1 = np.abs(np.fft.fft(gamma1))
        corr2 = np.abs(np.fft.fft(gamma2))
        if refl_bool:
            corr = np.maximum(corr1, corr2)
        else:
            corr = corr1
        refl = corr!=corr1
        
        ## Get the best angle for every image
        corr_max_ = np.max(corr,axis=1)
        idx_angle_max_ = np.argmax(corr,axis=1)
        ## Find K images that have the highest correlation
        # avoid sorting the full array
        idx_img = np.argpartition(-corr_max_, K,axis=0)[:K]
        # idx_img = idx_img[idx_img!=l]
        
        idx_best_img = idx_img[np.argsort(corr_max_[idx_img])[::-1]] # indices of the best match of images
        idx_angle_max = idx_angle_max_[idx_best_img] # indices of the angles corresponding to bets images
        corr_max = corr_max_[idx_best_img] # value of the correlation of the best images

        angle_est = 2*np.pi*idx_angle_max/corr.shape[1] # angle corresponding the the best matches in radian

        idx_best_img_knn[l] = idx_best_img
        angle_est_knn[l] = angle_est
        dist_knn[l] = 1-corr_max
        refl_knn[l] = refl[idx_best_img,idx_angle_max]

    return dist_knn, idx_best_img_knn, angle_est_knn, refl_knn



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
 - refl_bool: either or not compute reflection invariant distance

 OUTPUT:
  - dist_knn: 1-higest correlation
  - idx_best_img_knn: indeces of the K-NN best projections
  - angle_est_knn: angles of the K-NN best projections
  - refl_knn: boolean array indicating is the closest image is with or without reflection
"""
def rotation_invariant_knn_full(x1,K=10,ell_max=None,refl_bool=True):
    fb = ffb_2d.FFBBasis2D((x1.shape[1],x1.shape[1]),ell_max=ell_max,dtype=np.float64)
    c1 = fb.evaluate_t(x1)
    c1=fb.to_complex(c1)
    c1_normalized = c1/np.linalg.norm(c1,axis=1,keepdims=True)

    gamma1 = np.zeros((c1_normalized.shape[0],c1_normalized.shape[0],len(np.unique(fb.complex_angular_indices ))),dtype=complex)
    gamma2 = np.zeros((c1_normalized.shape[0],c1_normalized.shape[0],len(np.unique(fb.complex_angular_indices ))),dtype=complex)
    for kk in np.unique(fb.complex_angular_indices ):
        tmp = c1_normalized[:,fb.complex_angular_indices ==kk]
        gamma1[:,:,kk] = (np.conj(tmp[:,None,:])*tmp[None,:,:]).sum(2)
        gamma2[:,:,kk] = (tmp[:,None,:]*tmp[None,:,:]).sum(2)
    corr1 = np.abs(np.fft.fft(gamma1))
    corr2 = np.abs(np.fft.fft(gamma2))
    if refl_bool:
        corr = np.maximum(corr1, corr2)
    else:
        corr = corr1
    refl_est = np.max(corr,axis=-1) != np.max(corr1,axis=-1)
    ## Get the best angle for every image
    corr_max_ = np.max(corr,axis=-1)
    idx_angle_max_ = np.argmax(corr,axis=-1)
    from sklearn.neighbors import NearestNeighbors
    knn_distance_based = NearestNeighbors(n_neighbors=K, metric="precomputed").fit(corr_max_.max()-corr_max_)
    dist_knn, idx_best_img_knn = knn_distance_based.kneighbors(return_distance=True)
    angle_est = 2*np.pi*idx_angle_max_/corr.shape[-1]
    angle_est = np.array([angle_est[np.arange(idx_best_img_knn.shape[0]),idx_best_img_knn[:,k]] for k in range(K)]).T
    refl_est = np.array([refl_est[np.arange(idx_best_img_knn.shape[0]),idx_best_img_knn[:,k]] for k in range(K)]).T

    return dist_knn, idx_best_img_knn, angle_est, refl_est
