from utils.Graph import *
from utils.Data import *
from utils.Plotting import *
from utils.DiffusionMaps import *
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import fractional_matrix_power

########################### Parameters ###########################
N = 5
image_res = 100
snr = 10
K= 3

############################ Build up Simulation with projections and rotation invariant knn graph
aspire_vol, sim, clean_graph, noisy_graph = create_or_load_dataset_from_map("bunny-test-vdm", "bunny.map", N, image_res, snr, K, normalize=False, ctf=np.zeros((7)))
print(clean_graph.classes)

# currently we allow self-loops which should not be the case in VDM.

# Build Laplacian to compare against
A = create_adj_mat_nx(clean_graph.classes)
laplacian_embedding2 = calc_graph_laplacian(A, numberOfEvecs=3)
plot_3d_scatter(laplacian_embedding2)

# Build Diffusion Maps to compare agains
# Find good DM epsilon
epsilon_dm = 25
dists = euclidean_distances(A,A)
print(f"Distances: {dists.max()}, {dists.min()}, {dists.mean()}")
#aspire_angles_xyz = np.array([R.from_euler("ZYZ", angles).as_euler("XYZ") for angles in sim.angles])
x = np.array([ dist ** 2 / epsilon_dm for dist in dists])
y = np.array([ np.exp( - x_i) for x_i in x])
plt.scatter(x,y)
plt.title(f"Epsilon DM {epsilon_dm}")
plt.show()

# P = diffusion_map(X=A, alpha=epsilon_dm)
# #fig = plt.figure(figsize=(10, 10))
# D, psi=diffusion_distance(P, n_eign=3,t=100)

# print(f"psi: {psi.shape}")
# print(f"D: {D.shape}")
# coor1=(psi[:, 0]-psi[:, 0].mean())/psi[:, 0].std()
# coor2=(psi[:, 1]-psi[:, 1].mean())/psi[:, 1].std()
# coor3=(psi[:, 2]-psi[:, 2].mean())/psi[:, 2].std()
# plot_3dscatter(coor1, coor2, coor3)

# plot_3d_scatter(psi)
# plt.scatter(psi[:,0], coor2,cmap='hsv')


# Knn-Graph
# self.distance = distance
# self.classes = classes
# self.angles = angles
# self.reflection = reflection

# rotation between neighbours for optimal alignment

print(f"angles: shape: {clean_graph.angles.shape}, min: {clean_graph.angles.min()},  max: {clean_graph.angles.max()}, mean: {clean_graph.angles.mean()}")
print(clean_graph.angles)
print(f"distance: shape: {clean_graph.distance.shape}, min: {clean_graph.distance.min()},  max: {clean_graph.distance.max()}, mean: {clean_graph.distance.mean()}")
print(clean_graph.distance)

print(clean_graph.classes)

# angles: shape: (1000, 10), min: 0.0,  max: 6.240731352401346, mean: 2.8044403263229185
# within interval of [0, 2 pi]

# distance: shape: (1000, 10), min: -8.881784197001252e-16,  max: 0.010516059903895703, mean: 0.0025571909665922836
# normalizing distance?

# classes: graph edges
# reflection: bool indicating if neighbour is flected (image is flipped)



################################## Main VDM: ############################################

# Find good epsilon for VDM
epsilon_vdm = 0.0005

print(f"Distances: {clean_graph.distance.max()}, {clean_graph.distance.min()}, {clean_graph.distance.mean()}")
#aspire_angles_xyz = np.array([R.from_euler("ZYZ", angles).as_euler("XYZ") for angles in sim.angles])
x = np.array([ dist ** 2 / epsilon_vdm for dist in clean_graph.distance.flatten()])
y = np.array([ np.exp( - x_i) for x_i in x])
plt.scatter(x,y)
plt.title(f"Epsilon DM {epsilon_dm}")
plt.show()


# Build up Matrix S and D

dim = 2
O = np.zeros((N, K, dim, dim))
W = np.zeros((N, K, dim))
s_weights = np.zeros((N, K, dim, dim))
S = np.zeros((N * dim, N * dim))
D = np.zeros((N * dim, N * dim))
A_s = np.zeros((N,N))
degree = np.zeros(N)


# for all images
# calculate S, which includes W and O
for i in range(N):
  d_i = 0

  # iterate over all neighbours
  for j  in range(K):
    if clean_graph.classes[i,j] == i:
      A_s[i,i] = 1
      continue

    # optimal alignment angle and rotation
    theta = clean_graph.angles[i,j]
    c, s = np.cos(theta), np.sin(theta)
    r =  np.array(((c, -s), (s, c)))
    O[i,j] = r

    # weights including gaussian kernel
    # Equation 2.6
    w_ij = np.exp(- clean_graph.distance[i,j] ** 2 / epsilon_vdm)
    W[i,j] = w_ij

    # weighted alignment
    # Equation 3.1
    s_weights[i,j] = O[i,j] * w_ij

    # assign values to S
    d_i = d_i + w_ij # Equation 3.3
    row = i * dim
    col = clean_graph.classes[i,j] * dim
    S[row:row + dim, col : col+dim] = s_weights[i,j]
    A_s[i,clean_graph.classes[i,j]] = 1

  # set values of D
  row = i * dim
  D[row:row+dim, row:row+dim] = d_i * np.identity(dim) # Equation 3.2
  degree[i] = d_i


print(f"Weights: {W.shape}, max: {W.max()} , min: {W.min()}, mean: {W.mean()}")
print(W)
print(f"Transformations: {O.shape}, {O}")
print(f" data: {O}")
# print(f"Degrees: {degree}")

# print(D.shape)
print(f"D: {D}")
print('shape S:', S.shape)
print('S:', S)


# Compare A calculated by iteration and by ngx

# laplacian_embedding = calc_graph_laplacian(A_s, numberOfEvecs=3)
# plot_3d_scatter(laplacian_embedding)

# A = create_adj_mat_nx(clean_graph.classes)
# laplacian_embedding2 = calc_graph_laplacian(A, numberOfEvecs=3)
# plot_3d_scatter(laplacian_embedding2)


# print(f"A: {A}")
# print(f"As: {A_s}")


######################### Building S_tilde #####################
# Equation 3.12
D_0_5 = fractional_matrix_power(D, -0.5)
print(f"D ^-.5: {D_0_5}")

S_tilde = D_0_5 @ S @ D_0_5
print(f"S_tilde: {S_tilde}")


################# Spectral decomposition S_tilde ################
# Equation 3.13
t = 1
n_eign = 3

eign_values, eign_vecs = np.linalg.eig(S_tilde)

eign_values = eign_values.real
eign_vecs = eign_vecs.real
eign_value_idx = np.argsort(-np.abs(eign_values))

# sort decomposition by decreasing order of magnitude
eign_values = np.real(eign_values[eign_value_idx])
eign_vecs = np.real(eign_vecs[:,eign_value_idx])


print(f"e_val original shape: {eign_values.shape}") # dim * N
print(f"e_vec original shape: {eign_vecs.shape}") # (dim * N, dim * N)
print(eign_values)
print(eign_vecs)

# verify some staff
# decomposition of S_tilde and S_tilde ^ 2t
S_tilde_2t = fractional_matrix_power(S_tilde, 2 * t)


S_tilde_decomposed = np.zeros_like(S_tilde)
S_tilde_2t_decomposed = np.zeros_like(S_tilde_2t)

# calc decomposition
# Equation 3.13
for i in range (N):
  i_evec_idx = i * dim

  for j in range (N):
    if A_s[i,j] != 1 or i == j:
     continue

    j_evec_idx = j * dim
    res = np.zeros((dim, dim))
    res_2t = np.zeros_like(res)

    print("############################################################################################################")
    for l in range(N * dim):
      vl_i = eign_vecs[l][i_evec_idx: i_evec_idx + dim].reshape((dim, 1))
      vl_j = eign_vecs[l][j_evec_idx: j_evec_idx + dim].reshape((dim, 1))
      lambda_l = eign_values[l]

      res_tmp = (lambda_l * vl_i  * vl_j.T)

      res = res + res_tmp
      res_2t = res_2t + (lambda_l ** (2 * t) * vl_i * vl_j.T)

    row = i_evec_idx
    col = j_evec_idx
    print(f"result: ( {i}, {j}) = {res}")
    S_tilde_decomposed[row: row + dim, col : col+dim] = res
    S_tilde_2t_decomposed[row:row + dim, col : col+dim] = res_2t


print(f"S_tilde {S_tilde}")
print(f"S_tilde_decomposed {S_tilde_decomposed}")
print(f"S_tilde_2t {S_tilde_2t}")
print(f"S_tilde_2t_decomposed {S_tilde_2t_decomposed}")


# eign_values = np.real(eign_values[eign_value_idx[1:n_eign+1]])
# eign_vecs = np.real(eign_vecs[:,eign_value_idx[1:n_eign+1]])
# eign_values = np.real(eign_values[eign_value_idx])
# eign_vecs = np.real(eign_vecs[:,eign_value_idx])
# eign_values_t = eign_values ** (2*t)

################################# Vector diffusion mapping #################################
# Vector diffusion mapping
# complete one
# V_t shape :(N * dim, N * dim)
# # Equation 3.15

V_t = np.zeros((N, N * dim, N * dim))

for i in range (N):
  i_evec_idx = i * dim
  res = np.zeros((N * dim, N * dim))
  for l in range(N * dim):
    for r in range(N * dim):
      idx = l * N + r
      lambdas = (eign_values[l] * eign_values[r]) ** t 
      vl_i = eign_vecs[l][i_evec_idx: i_evec_idx + dim]
      vr_i = eign_vecs[r][i_evec_idx: i_evec_idx + dim]

      vt_lr = lambdas * np.dot(vl_i, vr_i)
      print (vt_lr)

      V_t[i,l, r] = vt_lr

# hs norm:
hs_norm_estimated = np.zeros((N,N))
hs_norm = np.zeros((N,N))
vdm_distance = np.zeros((N,N))
for i in range(N):
  for j in range(N):
    hs_norm[i,j] = np.linalg.norm(S_tilde_2t[i:i+dim, j:j+dim], ord='fro')
    hs_norm_estimated = np.dot(V_t[i], V_t[j])
    vdm_distance = euclidean_distances(V_t[i], V_t[j])


print(f" HS norm : {hs_norm}")
print(f" HS norm estimated : {hs_norm_estimated}")
print(f"vdm_distance: {vdm_distance}")


def S_ij(S, i, j, dim):
  return S[i:i+dim, j:j+dim]




assert False

input("Enter to terminate")