from vedo import Mesh,dataurl
from tqdm import tqdm
from skimage.transform import rotate
from scipy.spatial.transform import Rotation as R
import sys
sys.path.insert(0, '..')


mesh = Mesh(dataurl+"bunny.obj").normalize().subdivide()

mesh.show()

#pts = mesh.points(copy=True)  # pts is a copy of the points not a reference
pts = mesh.points()
pts[:,0]=pts[:,0]-(pts[:,0].max()+pts[:,0].min())/2
pts[:,1]=pts[:,1]-(pts[:,1].max()+pts[:,1].min())/2
pts[:,2]=pts[:,2]-(pts[:,2].max()+pts[:,2].min())/2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
import numpy as np

def get_2d(d3_points,resolution=100,xlim=[-2,2],ylim=[-2,2]):
    [xlim_l,xlim_u]=xlim
    [ylim_l,ylim_u]=ylim
    devx=(xlim_u-xlim_l)/resolution
    devy=(ylim_u-ylim_l)/resolution
    data_2d=d3_points[:,:2].T
    data_2d.shape[0]
    img=np.zeros((resolution,resolution))
    for i in range(data_2d.shape[1]):
        xi=int((data_2d[0,i]-xlim_l)/devx)
        yi=int((data_2d[1,i]-ylim_l)/devy)
        img[xi,yi]=img[xi,yi]+1
    return img

def rotation3d(d3_points,phi,psi,theta):
    r = R.from_euler('xyz', [phi, psi, theta], degrees=True)
    return r.apply(d3_points)

# from skimage import data
# x0 = data.shepp_logan_phantom()
x1=get_2d(rotation3d(pts,0,0,0))
x2=get_2d(rotation3d(pts,0,0,20))

# Check data are in a circle
coords = np.array(np.ogrid[:x1.shape[1], :x1.shape[1]],
                    dtype=object)
dist = ((coords - x1.shape[0] // 2) ** 2).sum(0)
outside_reconstruction_circle = dist > (x1.shape[0] // 2) ** 2
x1[outside_reconstruction_circle] = 0.
x2[outside_reconstruction_circle] = 0.
L=3000 # numer of images
x1 = x1[None,:,:]
x1 = x1.repeat(L,0)
for l in range(L):
    x1[l] = rotate(x1[0],360/L*l)


from scipy.spatial import distance_matrix
from FourierBessel import FBBasis2D
# from utils.FourierBessel import FBBasis2D
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


"""
from utils.FourierBessel import FBBasis2D
x1 = np.random.rand(10000,100,100)
ell_max = 5 # max order of the Fourier-Bessel approximation, will dictate the number of coeffs. See fb.count to see the number of coefficients
K = 10 # Nearest neighbors
M = 45 # number of angle to try

# If memory is an issue, or very high number of images
fb = FBBasis2D((x1.shape[1],x1.shape[1]),ell_max,dtype=np.float64)
dist, ind, angles = fb.Knn_mat(x1, K, M, verbose=False)

# If memory is not an issue, or very high number of images
dist2, angle2 = fb.distance_matrix(x1,x1,M)
nbrs = NearestNeighbors(n_neighbors=10, metric='precomputed').fit(np.abs(dist2))
distances, indices = nbrs.kneighbors()
"""


# Time test to compare GPU and CPU

import faiss
import time

nq = 100*90
d = 50
K = 10
u1 = np.random.random((nq, d)).astype('float32')
t = time.time()
index = faiss.IndexFlatL2(u1.shape[1])
index.add(u1.reshape(u1.shape[0],-1).copy(order='C').astype("float32"))
dist_, ind_ = index.search(u1.reshape(u1.shape[0],-1).copy(order='C').astype("float32"), K)
print("time: {0}".format(time.time()-t))

t = time.time()
res = faiss.StandardGpuResources()  # use a single GPU
index = faiss.IndexFlatL2(u1.shape[1])
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index_flat.add(u1.reshape(u1.shape[0],-1).copy(order='C').astype("float32"))
dist_, ind_ = gpu_index_flat.search(u1.reshape(u1.shape[0],-1).copy(order='C').astype("float32"), K)
print("time: {0}".format(time.time()-t))


