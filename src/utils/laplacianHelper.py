import numpy as np
# import cv2
from scipy.linalg import eigh
from utils.aspire_utils import rotation_invariant_knn
from scipy.spatial import distance_matrix


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


# Compute adjacency matrix with kNeighbours neighbours
def generateNeigbourHoodGraph(distanceMatrix, kNeighbours, sym=True):
	samples = distanceMatrix.shape[0]
	neighbourMatrix = np.zeros_like(distanceMatrix)
	for i in range(samples):
		distanceRow = distanceMatrix[i, :]
		index = np.argsort(distanceRow)
		neighbourMatrix[i, index[:kNeighbours+1]] = 1
		if sym:
			neighbourMatrix[index[:kNeighbours+1], i] = 1
	return neighbourMatrix


"""
dist: 'rotation' for rotatation invariant, otherwise standard l2 distance
"""
def computeLaplacianEmbedding(X, K, embedDim=3, epsilon=1e1, dist_str='l2', refl_bool=True, GL_invariant=False):
	# X_ = np.reshape(X,(X.shape[0],-1))
	# dist = euclidean_distances(X_,X_)
	print('Computing Distance...')
	# import ipdb; ipdb.set_trace()
	if dist_str == 'rotation':
		dist, ind, _, refl = rotation_invariant_knn(X,K=K,ell_max=None,refl_bool=refl_bool)
		neigbourMatrix = np.zeros((X.shape[0],X.shape[0]))
		distMatrix = np.zeros((X.shape[0],X.shape[0]))
		for k in range(X.shape[0]):
			neigbourMatrix[k,ind[k,:]] = 1
			distMatrix[k,ind[k,:]] = dist[k]
			neigbourMatrix[ind[k,:],k] = 1
			distMatrix[ind[k,:],k] = dist[k]
		affinMatrix = np.exp(-distMatrix**2/epsilon)*neigbourMatrix
		print('Distance Computed')
	elif dist_str == 'angle':
		dist = distance_matrix(X,X)
		dist = np.minimum(dist,360-dist)
	else:
		dist = distance_matrix(X,X)
	dist /= dist.max() # to make epsilon almost adimensional

	if dist_str != 'rotation':
		print('Distance Computed')
		neigbourMatrix = generateNeigbourHoodGraph(dist,K)
		neigbourMatrix = (((neigbourMatrix + neigbourMatrix.T)>0)*1).astype('float64')
		affinMatrix = np.exp(-dist**2/epsilon)

	if epsilon<=0:
		W = 1.*neigbourMatrix
	else:
		W = affinMatrix*neigbourMatrix

	if GL_invariant:
		D = W.sum(1)
		W_ = np.matmul(np.matmul(np.diag(1/D),W),np.diag(1/D))
		D_ = W_.sum(1)
		L = np.eye(X.shape[0])-np.matmul(np.diag(1/D_),W_)
	else:
		D = np.diag(np.sum(W,axis=1))
		L = D - W

	
	eVals, eVecs = eigh(L,) 
	embedding = eVecs[:,1:embedDim+1]

	return embedding, eVals, W
