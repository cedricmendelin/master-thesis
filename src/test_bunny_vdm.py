from utils.Graph import *
from utils.Data import *
from utils.Plotting import *
from utils.DiffusionMaps import *
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import fractional_matrix_power

N = 1000
image_res = 100
snr = 10
K=10

original_images, uniform_3d_angles, noisy_images, noise = create_or_load_bunny_immages(N, image_res, snr, save=True)

classes, reflections, rotations, correlations = create_or_load_knn_rotation_invariant(name='bunny', N=N, image_res=image_res, images=original_images, K=K)
classes_noisy, reflections_noisy, rotations_noisy, correlations_noisy = create_or_load_knn_rotation_invariant(name='bunny_noisy', N=N, image_res=image_res, images=noisy_images, K=K, snr=snr)

print("edges:" , np.count_nonzero(reflections))

print("rotation:")
# rotation between neighbours for optimal alignment
# within interval of [0, 2 pi]
print(rotations.shape) # (1000, 10)
print(rotations.min())
print(rotations.max())
print(rotations.mean())
# print(rotations)

print("correlations")
print(correlations.shape) # (1000, 10)
print(correlations.min()) # 10779.522965724176
print(correlations.max()) # 56478.243301824274
print(correlations.mean()) # 30753.471803816934

# normalize interval [0,1]
#correlations = (correlations - correlations.min()) / (correlations.max() - correlations.min())

alpha = 0.2

#dists = euclidean_distances(correlations,correlations)
#K = np.exp(-dists**2/alpha)

print("correlations")
print(correlations.shape) # (1000, 10)
print(correlations.min()) # 10779.522965724176
print(correlations.max()) # 56478.243301824274
print(correlations.mean()) # 30753.471803816934
# print(correlations)

# should we normalize it?
dim = 2
O = np.zeros((N, K, dim, dim))
s_weights = np.zeros((N, K, dim, dim))
S = np.zeros((N * dim, N * dim))
D = np.zeros((N * dim, N * dim))
A_s = np.zeros((N,N))

edge_count = 0

for i in range(N):
  d_i = 0
  
  
  for j  in range(K):
    # optimal alignment angle
    theta = rotations[i,j]
    c, s = np.cos(theta), np.sin(theta)
    # optimal alginment rotaiton
    r =  np.array(((c, -s), (s, c)))
    O[i,j] = r

    # weight
    # probably wrong, we need do put it in a gaussian kernel
    #w_i_j = correlations[i,j]
    w_i_j = np.exp(- correlations[i,j])
    

    # weighted alignment 
    s_weights[i,j] = O[i,j] * w_i_j

    #if reflections[i,j]:
    if True:
      edge_count = edge_count + 1
      d_i = d_i + w_i_j 
      row = i * dim
      col = classes[i,j] * dim
      S[row, col] = s_weights[i,j][0,0]
      S[row, col + 1] = s_weights[i,j][0,1]
      S[row + 1, col] = s_weights[i,j][1,0]
      S[row +1 , col + 1] = s_weights[i,j][1,1]

      A_s[i,classes[i,j]] = 1
      #print(i, classes[i,j])
  row = i * dim
  D[row:row+1, row:row+1] = d_i
  d_i = 0

print('shape S:', S.shape)


# S = S.reshape((N * dim, N* dim), order='F')
# D = D.reshape((N * dim, N* dim), order='F')

D_s = get_degree_matrix(A_s)

D_0_5 = fractional_matrix_power(D, -0.5)

S_tilde = D_0_5 @ S @ D_0_5

print('shape S_tilde:',S_tilde.shape)

t = 1
t_2 = 2 * t

S_tilde = np.nan_to_num(S_tilde)

S_tilde_power_2t = S_tilde ** t_2

print('shape S_tilde_power:', S_tilde_power_2t.shape)

print(np.isnan(S_tilde_power_2t).any())
print(np.isposinf(S_tilde_power_2t).any())
print(np.isneginf(S_tilde_power_2t).any())

S_2t_norm = np.linalg.norm(S_tilde_power_2t, ord='fro')
print('shape S_2t_norm:', S_2t_norm.shape)


A = create_adj_mat(classes, reflections)

print(np.array_equal(A, A_s))

print(A.shape)
print(A_s.shape)


n_eign = 3
eValues, eVectors = np.linalg.eig(S_tilde)

print(eValues)
eValues = eValues.real
eVectors = eVectors.real
eValueIndexOrder = np.argsort(-np.abs(eValues))

eValuesSorted = np.real(eValues[eValueIndexOrder[1:n_eign+1]])
eVectorsSorted = np.real(eVectors[:,eValueIndexOrder[1:n_eign+1]])
eValuesSortedT = eValuesSorted**t
psi   = eVectorsSorted @ np.diag(eValuesSortedT)

print(psi.shape)

plot_3dscatter(psi[:,0], psi[:,1], psi[:,2], (10,10))
print(psi.max())
print(psi.min())
print(psi.mean())


print(edge_count)

#A = create_adj_mat(classes, reflections)
#A_noisy = create_adj_mat(classes_noisy, reflections_noisy)
input("Enter to terminate")