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

aspire_vol, sim, clean_graph, noisy_graph = create_or_load_dataset_from_map("bunny-test", "bunny.map", N, image_res, snr, K, normalize=False, ctf=np.zeros((7)))

# Knn-Graoh
# self.distance = distance
# self.classes = classes
# self.angles = angles
# self.reflection = reflection

# rotation between neighbours for optimal alignment

print(f"angles: shape: {clean_graph.angles.shape}, min: {clean_graph.angles.min()},  max: {clean_graph.angles.max()}, mean: {clean_graph.angles.mean()}")
print(f"distance: shape: {clean_graph.distance.shape}, min: {clean_graph.distance.min()},  max: {clean_graph.distance.max()}, mean: {clean_graph.distance.mean()}")

# angles: shape: (1000, 10), min: 0.0,  max: 6.240731352401346, mean: 2.8044403263229185
# within interval of [0, 2 pi]

# distance: shape: (1000, 10), min: -8.881784197001252e-16,  max: 0.010516059903895703, mean: 0.0025571909665922836
# normalizing distance?

# classes: graph edges
# reflection: bool indicating if neighbour is flected (image is flipped)




epsilon = 0.2



# should we normalize it?
dim = 2
O = np.zeros((N, K, dim, dim))
s_weights = np.zeros((N, K, dim, dim))
S = np.zeros((N * dim, N * dim))
D = np.zeros((N * dim, N * dim))
A_s = np.zeros((N,N))


for i in range(N):
  d_i = 0
  
  
  for j  in range(K):
    # optimal alignment angle
    theta = clean_graph.angles[i,j]
    c, s = np.cos(theta), np.sin(theta)
    # optimal alginment rotaiton
    r =  np.array(((c, -s), (s, c)))
    O[i,j] = r

    # weight
    # probably wrong, we need do put it in a gaussian kernel
    #w_i_j = correlations[i,j]
    
    w_i_j = np.exp(- clean_graph.distance[i,j] ** 2 / epsilon)
    

    # weighted alignment 
    s_weights[i,j] = O[i,j] * w_i_j

    #if reflections[i,j]:
    if True:
      d_i = d_i + w_i_j 
      row = i * dim
      col = clean_graph.classes[i,j] * dim
      S[row, col] = s_weights[i,j][0,0]
      S[row, col + 1] = s_weights[i,j][0,1]
      S[row + 1, col] = s_weights[i,j][1,0]
      S[row +1 , col + 1] = s_weights[i,j][1,1]

      A_s[i,clean_graph.classes[i,j]] = 1
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


A = create_adj_mat(clean_graph.classes)

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