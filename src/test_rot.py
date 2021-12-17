from vedo import Mesh,dataurl
from tqdm import tqdm
from skimage.transform import rotate
from utils.Geometry import *

import sys
sys.path.insert(0, '..')


mesh = Mesh(dataurl+"bunny.obj").normalize().subdivide()

# mesh.show()

#pts = mesh.points(copy=True)  # pts is a copy of the points not a reference
pts = mesh.points()
pts[:,0]=pts[:,0]-(pts[:,0].max()+pts[:,0].min())/2
pts[:,1]=pts[:,1]-(pts[:,1].max()+pts[:,1].min())/2
pts[:,2]=pts[:,2]-(pts[:,2].max()+pts[:,2].min())/2

print(pts.shape)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import utils.plotting as plot
plt.ion()
import numpy as np



# from skimage import data
# x0 = data.shepp_logan_phantom()
x1=get_2d(rotation3d(pts,0,0,0))
# x2=get_2d(rotation3d(pts,0,0,20))


# Check data are in a circle
coords = np.array(np.ogrid[:x1.shape[1], :x1.shape[1]],
                    dtype=object)
dist = ((coords - x1.shape[0] // 2) ** 2).sum(0)
outside_reconstruction_circle = dist > (x1.shape[0] // 2) ** 2
x1[outside_reconstruction_circle] = 0.
#x2[outside_reconstruction_circle] = 0.
L=1000 # numer of images
x1 = x1[None,:,:]
x1 = x1.repeat(L,0)
for l in range(L):
    x1[l] = rotate(x1[0],360/L*l)


#plot.plot_imshow(x1[29])
#plot.plot_imshow(x1[0])
#plot.plot_imshow(x1[5])



from scipy.spatial import distance_matrix
from utils.FourierBessel import FBBasis2D
#from FourierBessel import FBBasis2D
fb = FBBasis2D((x1.shape[1],x1.shape[1]),ell_max=5,dtype=np.float64)
# dist, angle = fb.distance_matrix(x1,x1)


import time
M=360
# # Rotation with skimage
# t= time.time()
# dist1 = np.zeros((L,L))
# angles1 = np.zeros((L,L))
# # for l in range(L):
# for m in range(M):
#     tmp = np.zeros_like(x1)
#     for l in range(L):
#         tmp[l] = rotate(x1[l],360-360/M*m)
#     dist_ = np.abs(np.matmul(x1.reshape(L,-1),tmp.reshape(L,-1).T))
#     dist1 = np.maximum(dist_,dist1)
#     angles1[dist1==dist_] = 360/M*m
# print("Skimage        -- distance matrix: M={0} rotations, L={1} images. Elapsed time: {2} seconds.".format(M,L,time.time()-t))

# from sklearn.neighbors import NearestNeighbors
# t= time.time()
# dist2, angle2 = fb.distance_matrix(x1,x1,M)
# nbrs = NearestNeighbors(n_neighbors=10, metric='precomputed').fit(np.abs(dist2))
# distances, indices = nbrs.kneighbors()
# print("Fourier Bessel -- distance matrix: M={0} rotations, L={1} images. Elapsed time: {2} seconds.".format(M,L,time.time()-t))

# t = time.time()
# K=10
# dist, ind, angles = fb.Knn_mat(x1, K, M,False)
# print("Faiss KNN      -- distance matrix: M={0} rotations, L={1} images. Elapsed time: {2} seconds.".format(M,L,time.time()-t))

t = time.time()
K=10
dist2, ind2, angles2 = fb.Knn_mat_reduce_ram(x1, K, M,False)
print("Faiss KNN      -- distance matrix: M={0} rotations, L={1} images. Elapsed time: {2} seconds.".format(M,L,time.time()-t))
print(dist2.shape, ind2.shape, angles2.shape)


from utils.Graph import *
t = time.time()
A = create_adj_mat(ind2)
Dist = create_adj_mat(ind2, dist2)
Alpha = create_adj_mat(ind2, angles2)

epsilon = 1

W = calc_gaussian_kernel(Dist, epsilon)

# elementwise multiplication
S = np.multiply(W, Alpha)

print("Utils graph creation -- Elapsed time: {0} seconds.".format(time.time()-t))




from utils.DiffusionMaps import *

print(x1.shape)

#alpha=150

# Diffusion Maps
P=diffusion_map(X=A, alpha=1)
fig = plt.figure(figsize=(10, 10))
D, psi=diffusion_distance(P, n_eign=2,t=1)
coor1=(psi[:, 0]-psi[:, 0].mean())/psi[:, 0].std()
coor2=(psi[:, 1]-psi[:, 1].mean())/psi[:, 1].std()
plt.scatter(coor1, coor2,cmap='hsv')

# Vector Diffusion Maps:
from scipy.linalg import fractional_matrix_power
Degree_S = get_degree_matrix(S)
Degree_S_power_minus_05 = fractional_matrix_power(Degree_S, -0.5)
S_tilde = Degree_S_power_minus_05 @ S @ Degree_S_power_minus_05

eigenValues, eigenVectors = np.linalg.eigh(S_tilde)
idx = np.argsort(eigenValues)
fig = plt.figure(figsize=(10, 10))
#Plot with color to visualize the manifold.
plt.scatter(eigenVectors[:, idx[1]], eigenVectors[:, idx[2]],cmap='hsv')



# Laplacian:
L = np.diag(A.sum(axis=1)) - A
#Extract the second and third smallest eigenvector of the Laplace matrix.
eigenValues, eigenVectors = np.linalg.eigh(L)
idx = np.argsort(eigenValues)
fig = plt.figure(figsize=(10, 10))
#Plot with color to visualize the manifold.
plt.scatter(eigenVectors[:, idx[1]], eigenVectors[:, idx[2]],cmap='hsv')


#print(A)
#print(W)
#print(Theta)
"""
from utils.FourierBessel import FBBasis2D
x1 = np.random.rand(10000,100,100)
ell_max = 5 # max order of the Fourier-Bessel approximation, will dictate the number of coeffs. See fb.count to see the number of coefficients
K = 10 # Nearest neighbors
M = 45 # number of angle to try

# If memory is an issue, or very high number of images
fb = FBBasis2D((x1.shape[1],x1.shape[1]),ell_max,dtype=np.float64)
dist, ind, angles = ppip

# If memory is not an issue, or very high number of images
dist2, angle2 = fb.distance_matrix(x1,x1,M)
nbrs = NearestNeighbors(n_neighbors=10, metric='precomputed').fit(np.abs(dist2))
distances, indices = nbrs.kneighbors()
"""


input("Enter to terminate")

# Time test to compare GPU and CPU
