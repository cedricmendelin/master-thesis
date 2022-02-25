from utils.Graph import *
from utils.Data import *
from utils.Plotting import *
from utils.DiffusionMaps import *
from utils.VectorDiffusionMaps import *
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import fractional_matrix_power

########################### Parameters ###########################
N = 10
image_res = 20
snr = 10
K = 4

############################ Build up Simulation with projections and rotation invariant knn graph
aspire_vol, sim, clean_graph, noisy_graph = create_or_load_dataset_from_map("bunny-test-vdm", "bunny.map", N, image_res, snr, K, normalize=False, ctf=np.zeros((7)))

# currently we allow self-loops which should not be the case in VDM.

# Build Laplacian to compare against
# A = create_adj_mat_nx(clean_graph.classes)
# laplacian_embedding = calc_graph_laplacian(A, numberOfEvecs=3)
# plot_3d_scatter(laplacian_embedding)

# A_noisy = create_adj_mat_nx(noisy_graph.classes)
# laplacian_embedding2 = calc_graph_laplacian(A, numberOfEvecs=3)
# plot_3d_scatter(laplacian_embedding2)


# Build Diffusion Maps to compare agains
# Find good DM epsilon
# epsilon_dm = 25
# dists = euclidean_distances(A,A)
# print(f"Distances: {dists.max()}, {dists.min()}, {dists.mean()}")
# #aspire_angles_xyz = np.array([R.from_euler("ZYZ", angles).as_euler("XYZ") for angles in sim.angles])
# x = np.array([ dist ** 2 / epsilon_dm for dist in dists])
# y = np.array([ np.exp( - x_i) for x_i in x])
# plt.scatter(x,y)
# plt.title(f"Epsilon DM {epsilon_dm}")
# plt.show()

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

# print(f"angles: shape: {clean_graph.angles.shape}, min: {clean_graph.angles.min()},  max: {clean_graph.angles.max()}, mean: {clean_graph.angles.mean()}")
# print(clean_graph.angles)
# print(f"distance: shape: {clean_graph.distance.shape}, min: {clean_graph.distance.min()},  max: {clean_graph.distance.max()}, mean: {clean_graph.distance.mean()}")
# print(clean_graph.distance)

# print(clean_graph.classes)

# angles: shape: (1000, 10), min: 0.0,  max: 6.240731352401346, mean: 2.8044403263229185
# within interval of [0, 2 pi]

# distance: shape: (1000, 10), min: -8.881784197001252e-16,  max: 0.010516059903895703, mean: 0.0025571909665922836
# normalizing distance?

# classes: graph edges
# reflection: bool indicating if neighbour is flected (image is flipped)



################################## Main VDM: ############################################

# Find good epsilon for VDM
# 10 / 4 : 0.05
epsilon_vdm = 0.00008
verify_epsilon(noisy_graph.distance, [0.0005, 0.00001, 0.00003, 0.00005, 0.00008, 0.000005])


# other Parameters:
dim = 2
sym = True
t = 50
t_2 = 2 * t
n_eign = 64


######################### Build up Matrix S and D #####################

S, D, A_s = calculate_S(N, K, noisy_graph, epsilon_vdm, dim, sym)

######################### Building S_tilde #####################
# Equation 3.12
D_0_5 = fractional_matrix_power(D, -0.5)
# print(f"D ^-.5: {D_0_5}")

S_tilde = D_0_5 @ S @ D_0_5
# print(f"S_tilde: {S_tilde}")

################# Spectral decomposition S_tilde ################
# Equation 3.13
eign_values, eign_vecs = np.linalg.eigh(S_tilde)
eign_value_idx = np.argsort(-np.abs(eign_values))

# verfiy_spectral_decomposition(eign_values, eign_vecs, eign_value_idx, S_tilde, n_eign_start=10, n_eign_stop=54, n_eign_step=4)

# print(f"S_tilde_decomposed {S_tilde_decomposed}")
#eign_values = eign_values.real
#eign_vecs = eign_vecs.real

eign_values_org = eign_values
eign_vecs_org = eign_vecs


# # sort decomposition by decreasing order of magnitude
eign_values = eign_values[eign_value_idx[0:n_eign+1]]
eign_vecs = eign_vecs[:,eign_value_idx[0:n_eign+1]]

S_tilde_t2_decomposed = (eign_vecs[:,None,:] * eign_vecs[None,:,:] * eign_values[None,None,:] ** t_2).sum(2)
S_tilde_t2 = fractional_matrix_power(S_tilde, t_2)


diff_2t_tilde = np.linalg.norm(S_tilde_t2 - S_tilde_t2_decomposed) / np.linalg.norm(S_tilde_t2)

print(f"Diff spectral decomposition 2t S_tilde: {diff_2t_tilde}")


# print(eign_values ** 10)
# print(eign_vecs)


# plt.hist(eign_values)
# plt.show()

# ignore decomposing:
S_tilde_t2_decomposed = S_tilde_t2

################################# Vector diffusion mapping #################################
# Vector diffusion mapping
# complete one
# V_t shape :(N * dim, N * dim)
# # Equation 3.15

# hs norm:
hs_norm = calculate_hs_norm(S_tilde_t2, N, dim)
hs_norm_decomposed = calculate_hs_norm(S_tilde_t2_decomposed, N, dim)

vdm_distance = calculate_vdm_distance(hs_norm, N)
vdm_distance_decomposed = calculate_vdm_distance(hs_norm_decomposed, N)

diff_hs_norm = np.linalg.norm(hs_norm - hs_norm_decomposed) / np.linalg.norm(hs_norm)
diff_vdm_distance = np.linalg.norm(vdm_distance - vdm_distance_decomposed) / np.linalg.norm(vdm_distance)

print(f"Diff hs norm: {diff_hs_norm}")
print(f"Diff vdm distance: {diff_vdm_distance}")

# print(f"hs norm {hs_norm}")
# print(f"vdm distance {vdm_distance}")



vdm_graph, classes = generate_knn_from_distances(vdm_distance, K, ordering='desc')
# vdm_graph_laplacian = calc_graph_laplacian_nx(vdm_graph, embedDim=3)

# plot_3d_scatter(vdm_graph_laplacian, title=f"VDM with t={t}")


G_estimated = nx.convert_matrix.from_numpy_matrix(vdm_graph)

G_true = nx.convert_matrix.from_numpy_matrix(create_adj_mat(clean_graph.classes) - np.identity(A_s.shape[0]))
G_noisy_true = nx.convert_matrix.from_numpy_matrix(create_adj_mat(noisy_graph.classes) - np.identity(A_s.shape[0]))

print(f"recovered edges: {G_estimated.edges}, # {len(G_estimated.edges)}")
print(f"true edges: {G_true.edges}, # {len(G_true.edges)}")
print(f"true noisy edges: {G_noisy_true.edges}, # {len(G_noisy_true.edges)}")

# symmetric_differece: set of all the elements that are either in the first set or the second set but not in both.
# difference: set of all the elements in first set that are not present in the second set
# intersection: set of all the common elements of both the sets
# union: set of all the elements of both the sets without duplicates

set_diff_noisy = set(G_noisy_true.edges).symmetric_difference(set(G_estimated.edges))
set_diff_true = set(G_true.edges).symmetric_difference(set(G_estimated.edges))
set_diff_true_noisy = set(G_true.edges).symmetric_difference(set(G_noisy_true.edges))

print(f"edges diff true and noisy: {set_diff_noisy}, # {len(set_diff_noisy)}")
print(f"edges diff estimated noisy: {set_diff_noisy}, # {len(set_diff_noisy)}")
print(f"edges diff estimated true: {set_diff_true}, # {len(set_diff_true)}")

print(f"clean classes {clean_graph.classes}")
print(f"noisy classes {noisy_graph.classes}")

# print(f"clean A,  {create_adj_mat(clean_graph.classes)} ")
# print(f"noisy A,  {create_adj_mat(noisy_graph.classes)} ")
# print(f"A_S: {A_s}")

# vdm_graph_laplacian = calc_graph_laplacian_nx(vdm_graph, embedDim=3)
# print(vdm_graph_laplacian.shape)
# plot_3d_scatter(vdm_graph_laplacian)


def S_ij(S, i, j, dim):
  ii = i * dim
  jj = j * dim
  return S[ii:ii+dim, jj:jj+dim]




assert False

input("Enter to terminate")