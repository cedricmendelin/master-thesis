# Simple example to validate first implementation:
# generate out of plane rotation by discretizing the sphere using icosphere and extract the two corresponding angles. Let say you have N angles
# Generate in plane rotation, uniformly distributed between [0,2pi], let say 20 rotations per out of plane rotation
# Then you get you 3 angles
# Do the rotation (be careful to apply inplane rotation at the end, so that it is truly inplane), you have N*20 projections
# Compute the N rotation invariant distances,. Distance between image i and image j: compute l2 distance between i and all 10 inplane rotations of j

# To get the uniform discretization of the sphere
from json import load
from turtle import distance
from icosphere import icosphere
import numpy as np
from utils.Plotting import *
from utils.Graph import *
from utils.DiffusionMaps import *
from utils.VectorDiffusionMaps import *


from utils.Normalization import normalize_min_max
# from utils.Geometry import *
from scipy.spatial.transform import Rotation as R
from numpy.random import default_rng
from scipy.ndimage.interpolation import rotate
from skimage.transform import resize
import mrcfile

from vedo import dataurl, Mesh, mesh2Volume, volumeFromMesh
from tqdm import tqdm
import os.path as path

def rotate_volume(V, alpha, beta):
    alpha = np.rad2deg(alpha)
    beta = np.rad2deg(beta)

    # (2,1) -> x-axis rotation
    # (2,0) -> y-axis rotation
    # (1,0) -> z-axis rotation
    V = rotate(V, alpha, mode='constant', cval=0, order=3, axes=(2, 1), reshape=False)
    V = rotate(V, beta, mode='constant', cval=0, order=3, axes=(2, 0), reshape=False)
    return V

def cart2sph(x,y,z):
    r=np.sqrt(x**2+y**2)+1e-16
    theta_=np.arctan(z/r)
    phi_=np.arctan2(y,x)
    return phi_, theta_
    
def sph2cart(phi,theta):
    #takes list rthetaphi (single coord)
    x = np.cos( theta ) * np.cos( phi )
    y = np.cos( theta ) * np.sin( phi )
    z = np.sin( theta )
    return x,y,z

def getPSF(sigma,n):
    if sigma==0:
        return 0, 0, 0
    else:
        ext = 3*sigma # padding size
        lin = np.linspace(-n/2,n/2,n)
        XX,YY = np.meshgrid(lin,lin)
        if sigma==0:
            g = np.zeros((n,n))
            g[n//2,n//2]=1
        else:
            g = np.exp(-(XX**2+YY**2)/(2*sigma**2))
            g /= np.sum(g)

        g_hat = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g)))
        g = np.repeat(g[:,:,None],n,axis=2)
        g_hat = np.repeat(g_hat[:,:,None],n,axis=2)
        return g, g_hat, ext

def getRotationMatrix_t(angles,order='ZYX'):
    rr = R.from_euler(order,angles.detach().cpu().numpy(),degrees=True)
    mat = rr.as_matrix()
    rotationMatrix = torch.tensor(mat).type(torch.float).to(angles.device)
    return rotationMatrix

def S_ij(S, i, j, dim=2):
  ii = i * dim
  jj = j * dim
  return S[ii:ii+dim, jj:jj+dim]


from utils.generateHelper import forward_t, find_sigma_noise_t

################### Parameters #################
# defines number of rotations points
nu = 6  # or any other integer (10 leads to 1002 points, 4 to 162)
N_inplane = 8 # total with 4,8 : 1296
resolution = 50
K=6
recreate_data = True
debug = False

################### Generete out of plane rotation (2 angles) ###################
vertices, faces = icosphere(nu)
ax1,ax2,ax3=vertices[:,0],vertices[:,1],vertices[:,2]
rot_x, rot_y = cart2sph(ax1,ax2,ax3)

N = vertices.shape[0]

# vertices: shape (1002, 3)
# rotations shape (1002)
# faces (2000, 3)
################### Generate in plane rotation ###################

N_projections = N * N_inplane

rng = default_rng()
rot_z = np.linspace(0,2*np.pi, N_inplane)
# rot_z = rng.uniform(0, 2 * np.pi, N_inplane)

# map everything together
rot = np.zeros((N_projections, 3))
for i in range(N):
    for j in range(N_inplane):
        rot[i * N_inplane + j] = np.array([rot_x[i],rot_y[i],rot_z[j]])

rot = rot*180/(np.pi)


################## Load Volume ################

map_file_path = "src/maps/padding_bunny.map"
vol_org = mrcfile.open(map_file_path).data.astype(np.float64)
vol_org = resize(vol_org, (resolution, resolution, resolution))
# V=np.load('./toyModel/FLATFOOT_StanfordBunny_jmil_HIGH_RES_Smoothed_VOXELSZ_0.02_SZ_100.npy')
# V=resize(V,(resolution,resolution,resolution))
# vol_org=V
################# Generate projections #############
# file = "test_padding_bunny_projections.npz"
# projections = np.zeros((N_projections, resolution, resolution))
# images = np.zeros((N, resolution, resolution))
# angles = np.zeros((N_projections, 3))

# Valentin approach:
import torch
np.random.seed(0)
torch_type=torch.float
use_cuda=torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("GPU: ", use_cuda)

# next to angle, we need shifts
#Generate Angles and shifts
thetaSet = rot
Ntheta = N_projections
n = resolution
sigma_psf = 0
shiftWidth=0
SNR_value=5
cuda=True

g,_,paddingSize = getPSF(sigma_psf,n)
shifts = np.random.random(size=(Ntheta,2))*shiftWidth
shifts = np.zeros((Ntheta, 2))


# If possible, use cuda and pytorch to fasten the data generatin
if use_cuda and cuda:
    # Convert numpy to cuda torch
    thetaSet_t = torch.tensor(thetaSet.copy()).type(torch_type).to(device)
    g_t = torch.tensor(g).type(torch_type).to(device)
    V_t = torch.FloatTensor(vol_org).unsqueeze(0).type(torch_type).to(device)
    ## Generate an experiments
    P_t = torch.zeros((Ntheta,n,n)).type(torch_type).to(device)
    P_t_ = torch.zeros((Ntheta,n,n)).type(torch_type).to(device)
    angleMatrixTorch = getRotationMatrix_t(thetaSet_t,'XYZ').type(torch_type).to(device)

    for k in tqdm(range(Ntheta), "Generating projections with CUDA"):
        P0 = forward_t(V_t, angleMatrixTorch[k], shifts[k],g_t,paddingSize)
        sigma_noise = find_sigma_noise_t(SNR_value,P0)
        y = P0 + torch.randn_like(P0)*sigma_noise
        P_t[k] = y
        P_t_[k] = P0
    P = P_t.detach().cpu().numpy()
    P_ = P_t_.detach().cpu().numpy()

else:
    assert False , "No Cuda available"

projections = P_
noisy_projections = P

# plot some projections
if debug:
  w = 10
  h = 10
  fig = plt.figure(figsize=(8, 8))
  columns = 4
  rows = 4
  for i in range(1, columns*rows +1):
      ax = fig.add_subplot(rows, columns, i)
      ax.set_title(f"img {i}")
      plt.imshow(projections[i-1])
  plt.show()

############### Calculate distances ###################
# Compute the N rotation invariant distances,. 
# Distance between image i and image j: compute l2 distance between i and all 10 inplane rotations of j
tmp_projections = projections.reshape(N, N_inplane, resolution, resolution)
tmp_angles = np.deg2rad(rot[:,2].reshape(N, N_inplane))
distances = np.zeros((N, N))
angles = np.zeros((N, N))

for i in tqdm(range(N), "Computing distances"):
  for j in range(N):
    ref_dist = np.zeros(N_inplane)
    for p in range(N_inplane):
      ref_dist[p] = np.linalg.norm(tmp_projections[i, 0] - tmp_projections[j, p])
    idx_min = ref_dist.argmin()
    distances[i,j] = ref_dist[idx_min]
    angles[i,j] = tmp_angles[j, idx_min]

# normalize_distances:
distances /= distances.max()

################ Creating graph ###########################
from utils.Data import Knn
graph, classes = generate_knn_from_distances(distances, K, ordering='asc', ignoreFirst=True)

angle_per_node = np.zeros_like(classes, dtype=np.float)
distance_per_node = np.zeros_like(classes, dtype=np.float)
for i in range(N):
  for j in range(K):
    c_idx = classes[i,j]
    angle_per_node[i,j] = angles[i, c_idx]
    distance_per_node[i,j] = distances[i,c_idx]

# print(f"Should be zero {np.linalg.norm(graph - graph.T)}")

print(f"angles: {angles}")
# print(f"distances: {distances}")
# print(f"angles per node: {angle_per_node}")
# print(f"distances per node: {distance_per_node}")

# graph=0.5*(graph+graph.T)
# graph[graph>1]=1



################### Graph Laplacian #####################
from scipy import sparse

graph_laplacian = calc_graph_laplacian_nx(graph, embedDim=3)
# graph_laplacian=sparse.csr_matrix(graph_laplacian)
plot_3d_scatter(graph_laplacian, title=f"Graph Laplacian input graph")


#################### Diffusion Maps #########################
# t = 2
# W = graph * distances
# epsilon_dm = 0.1

# verify_epsilon(W, [0.0001, 0.001, 0.01, 0.1, 0.2])

# P = diffusion_map(X=W, alpha=epsilon_dm)

# for t in range (0, 100, 5):
#   D, psi=diffusion_distance(P, n_eign=362,t=t)
#   graph_dm, classes_dm = generate_knn_from_distances(D, K, ordering='asc')

#   graph_laplacian_dm = calc_graph_laplacian_nx(graph_dm, embedDim=3)
#   # graph_laplacian=sparse.csr_matrix(graph_laplacian)
#   plot_3d_scatter(graph_laplacian_dm, title=f"Graph Laplacian DM t={t}")

plt.show()

assert False

######################### VDM ###############################
from utils.VectorDiffusionMaps import *
clean_graph = Knn(distance_per_node, classes, angle_per_node, None)

epsilon_vdm = 0.7
# verify_epsilon(clean_graph.distance, [20000, 40000, 60000, 80000, 90000])
verify_epsilon(distances, [0.25, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9])

# other Parameters:
dim = 2
sym = True
t = 1
t_2 = 2 * t
n_eign = N * 2

S, D, A_s = calculate_S(N, K, clean_graph, epsilon_vdm, dim, sym)

# Equation 3.12
D_0_5 = fractional_matrix_power(D, -0.5)
# print(f"D ^-.5: {D_0_5}")

S_tilde = D_0_5 @ S @ D_0_5

# Equation 3.13
eign_values, eign_vecs = np.linalg.eigh(S_tilde)
eign_value_idx = np.argsort(-np.abs(eign_values))

# verfiy_spectral_decomposition(eign_values, eign_vecs, eign_value_idx, S_tilde, n_eign_start=10, n_eign_stop=54, n_eign_step=4)

# sort decomposition by decreasing order of magnitude
# eign_values_org = eign_values
# eign_vecs_org = eign_vecs
eign_values = eign_values[eign_value_idx[0:n_eign+1]]
eign_vecs = eign_vecs[:,eign_value_idx[0:n_eign+1]]

#S_tilde_t2_decomposed = (eign_vecs[:,None,:] * eign_vecs[None,:,:] * eign_values[None,None,:] ** t_2).sum(2)

# for t in range (1, 40, 2):

S_tilde_t2 = np.linalg.matrix_power(S_tilde, t_2) # fractional_matrix_power

hs_norm_S = calculate_hs_norm(S_tilde_t2, N, dim)
vdm_distance_S = calculate_vdm_distance(hs_norm_S, N)
vdm_graph_S, vdm_classes_S = generate_knn_from_distances(vdm_distance_S, K, ordering='asc', ignoreFirst=True)
print(f"try understand something norm: {hs_norm_S}")
print(f"try understand something distance: {vdm_distance_S}")
print(f"vdm S classes {vdm_classes_S}")
print(f"evals: {eign_values}")


hs_norm_trace = np.zeros_like(hs_norm_S)
hs_norm_evals = np.zeros_like(hs_norm_S)
for i in range(N):
  for j in range(N):
    hs_norm_trace[i,j] = np.trace(S_ij(S_tilde_t2, i,j) @ S_ij(S_tilde_t2, i,j).T)
    res = 0
    ii = i*dim
    jj = j*dim
    for l in range(N*dim):
      for r in range(N*dim):
        t = ((eign_values[l] * eign_values[r]) ** t_2) * np.dot(eign_vecs[l, ii:ii+dim], eign_vecs[r, ii:ii+dim]) * np.dot(eign_vecs[l, jj:jj+dim], eign_vecs[r, jj:jj+dim])
        res = res + t
    hs_norm_evals[i,j] = res


print(f"hs norm with Trace: {hs_norm_trace}")
print(f"Should be zero: {np.linalg.norm(hs_norm_S- hs_norm_trace)}")

print(f"hs norm from evals: {hs_norm_evals}")
print(f"Should be zero: {np.linalg.norm(hs_norm_S- hs_norm_evals)}")



#diff_2t_tilde = np.linalg.norm(S_tilde_t2 - S_tilde_t2_decomposed) / np.linalg.norm(S_tilde_t2)

#print(f"Diff spectral decomposition 2t S_tilde: {diff_2t_tilde}")

################################# Vector diffusion mapping #################################
# Vector diffusion mapping
# complete one
# V_t shape :(N * dim, N * dim)
# # Equation 3.15

# hs norm:
hs_norm = calculate_hs_norm(S_tilde_t2, N, dim)
hs_norm_from_eign = np.zeros_like(hs_norm)

# for i in range(N):
#   for j in range(N):





# hs_norm_decomposed = calculate_hs_norm(S_tilde_t2_decomposed, N, dim)

vdm_distance = calculate_vdm_distance(hs_norm, N)
# vdm_distance_decomposed = calculate_vdm_distance(hs_norm_decomposed, N)


# diff_hs_norm = np.linalg.norm(hs_norm - hs_norm_decomposed) / np.linalg.norm(hs_norm)
# diff_vdm_distance = np.linalg.norm(vdm_distance - vdm_distance_decomposed) / np.linalg.norm(vdm_distance)

# print(f"Diff hs norm: {diff_hs_norm}")
# print(f"Diff vdm distance: {diff_vdm_distance}")

# print(f"S: {S}")
# print(f"S_tilde: {S_tilde}")
# print(f"S_tilde_t2: {S_tilde_t2}")

# print(f"hs_norm: {hs_norm}")

# print(f"original distances: {distances}")
# print(f"vdm distances: {vdm_distance}")

vdm_graph, vdm_classes = generate_knn_from_distances(vdm_distance, K, ordering='desc')
vdm_graph_laplacian = calc_graph_laplacian_nx(vdm_graph, embedDim=3)

plot_3d_scatter(vdm_graph_laplacian, title=f"Laplacian VDM t={t}")

# comparing graphs:


G_estimated = nx.convert_matrix.from_numpy_matrix(vdm_graph)

G_true = nx.convert_matrix.from_numpy_matrix(graph)

print(f"recovered edges: {G_estimated.edges}, # {len(G_estimated.edges)}")
print(f"true edges: {G_true.edges}, # {len(G_true.edges)}")

# symmetric_differece: set of all the elements that are either in the first set or the second set but not in both.
# difference: set of all the elements in first set that are not present in the second set
# intersection: set of all the common elements of both the sets
# union: set of all the elements of both the sets without duplicates

set_diff_true = set(G_true.edges).symmetric_difference(set(G_estimated.edges))
print(f"edges diff estimated true: {set_diff_true}, # {len(set_diff_true)}")

print(f"clean classes {classes}")
print(f"estimated classes {vdm_classes}")

# save to csv file

np.savetxt('distances.csv', normalize_min_max(distances), delimiter=',', fmt='%.2f')
np.savetxt('vdm_distances.csv', normalize_min_max(vdm_distance), delimiter=',', fmt='%.2f')

plt.show()