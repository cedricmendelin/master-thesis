import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import default_rng

from utils.Graph import *
from utils.Plotting import *
from utils.VectorDiffusionMaps import *
from utils.DiffusionMaps import *
from utils.Normalization import *

np.set_printoptions(precision=4)

# Parameters
N = 100
K = 76
debug_plot = True
debug_print = False
calculate_DM = False
calculate_VDM = True

############### Sampling points on circle ###########################

step = 2*np.pi / N
angle_in_radian = np.linspace(0, 2*np.pi - step,N)
point_on_cricle = np.array([np.cos(angle_in_radian), np.sin(angle_in_radian)]).T

if debug_plot:
  plt.figure()
  plt.scatter(point_on_cricle[:,0], point_on_cricle[:,1], s=20)
  plt.xticks([-1, -0.5,0,0.5,1])
  plt.yticks([-1, -0.5,0,0.5,1])

if debug_print:
  print(f"Points on circle: {point_on_cricle}")
  print(f"Angles in radian: {angle_in_radian}")

################# Computing distances #########################

def great_circle_distance(lon1, lat1, lon2, lat2, debug_i):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # if debug_i == 9:
    #   t1 = np.sin(lat1) 
    #   t2 = np.sin(lat2) 
    #   t3 = np.cos(lat1)
    #   t4 = np.cos(lat2)
    #   t5 = np.cos(lon1 - lon2)
    #   arccos_term = t1 * t2 + t3 * t4 * t5
    #   t_arccos = np.arccos(max(min(arccos_term, 1), -1))
    #   if np.isnan(t_arccos):
    #     print(f"t1 {t1}, t2 {t2}, t3 {t3}, t4 {t4}, t5 {t5}")
    #     print(f"lon1 {lon1}, lon2 {lon2}, lat1 {lat1}, lat2 {lat2}")
    #     print(f"arccos_term: {arccos_term} ,result: {t_arccos}")
    #   return t_arccos
    # else:
    return np.arccos( max(min(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2), 1), -1))

distances = np.array([great_circle_distance(point_on_cricle[i,0], point_on_cricle[i,1], point_on_cricle[j,0], point_on_cricle[j,1], i) for i in range(N) for j in range(N)]).reshape((N,N))

distances /= distances.max()

if debug_print:
  print(f"Distances: {distances}")
  print(f"Distances: {distances.max()}")
  print(f"Distances: {distances.min()}")
  print(f"Distances: {distances.mean()}")


print("Distances done!")

############### Generating Knn graph #########################

graph, classes = generate_knn_from_distances(distances, K, ordering='asc', ignoreFirst=True)

if debug_plot:
  graph_laplacian = calc_graph_laplacian_nx(graph, embedDim=2)
  plot_2d_scatter(graph_laplacian, title=f"Graph Laplacian input graph")

if debug_print:
  print("Graph: ", graph)
  print("Neighbours", classes)

print("Input Graph done!")

#################### Diffusion Maps #########################
if calculate_DM:
  W = graph * distances

  epsilon_dm = 0.5 #distances.std() ** 2

  # verify_epsilon(distances, [1, 0.5, distances.std() ** 2])


  P = diffusion_map(X=W, alpha=epsilon_dm)
  graph_laplacian_dm = calc_graph_laplacian_nx(P, embedDim=2)
  plot_2d_scatter(graph_laplacian_dm, title=f"Graph Laplacian DM: {N}, K={K}")

  


  distances_dm_geodesic = np.array([great_circle_distance(graph_laplacian_dm[i,0], graph_laplacian_dm[i,1], graph_laplacian_dm[j,0], graph_laplacian_dm[j,1], i) for i in range(N) for j in range(N)]).reshape((N,N))
  distances_dm_geodesic /= distances_dm_geodesic.max()
  graph_dm, classes_dm = generate_knn_from_distances(distances_dm_geodesic, K, ordering='asc', ignoreFirst=True)

  graph_laplacian_dm2 = calc_graph_laplacian_nx(graph_dm, embedDim=2)

  fig = plt.figure()
  plt.scatter(graph_laplacian_dm2[:,1], np.zeros(N), cmap='hsv')
  plt.title(f"Graph Laplacian DM, geodesic distances circle: {N}, K={K}")

  circle = normalize_range(np.abs(graph_laplacian_dm2[:,1]), 0, 2 * np.pi)
  data = np.array([np.sin(circle), np.cos(circle) ]).T
  plot_2d_scatter(data, title=f"Graph Laplacian DM, geodesic distances circle: {N}, K={K}")


############################ Vector Diffusion Maps ##############################
from utils.VectorDiffusionMaps import *
from scipy.sparse.linalg import eigsh
from utils.Data import Knn
from scipy import sparse

if calculate_VDM:
  S = np.zeros((N*2,N*2))
  Dinv = np.zeros((N*2,N*2))

  for i in range(N):
      for j in range(i+1,N):
          #theta = angles[i,j]/180*np.pi
          theta = 0
          c, s = np.cos(theta), np.sin(theta)
          R = np.array(((c, -s), (s, c)))
          S[i*2:i*2+2,j*2:j*2+2] = graph[i,j]*R
          S[j*2:j*2+2,i*2:i*2+2] = graph[i,j]*R.T
      Dinv[i,i] = 1/np.sum(graph[i])
      Dinv[i*2,i*2] = 1/np.sum(graph[i])
      Dinv[i*2+1,i*2+1] = 1/np.sum(graph[i])

  D_inv_half = np.sqrt(Dinv)
  Stilde = np.matmul(np.matmul(D_inv_half,S),D_inv_half)

  t = 2
  Stilde_t = np.linalg.matrix_power(Stilde,t)

  aff = np.zeros((N,N))
  for i in range(N):
      for j in range(i+1,N):
          aff[i,j] = np.linalg.norm(Stilde_t[i*2:i*2+2,j*2:j*2+2],'fro')**2
          aff[j,i] = np.linalg.norm(Stilde_t[i*2:i*2+2,j*2:j*2+2],'fro')**2

  graph_laplacian_vdm_Stilde = calc_graph_laplacian_nx(Stilde, embedDim=2)
  plot_2d_scatter(graph_laplacian_vdm_Stilde, title=f"Graph Laplacian VDM S_tilde: {N}, K={K}, t={t/2}")

  graph_laplacian_vdm_aff = calc_graph_laplacian_nx(aff, embedDim=2)
  plot_2d_scatter(graph_laplacian_vdm_aff,title=f"Graph Laplacian VDM affinity: {N}, K={K}, t={t/2}")


  # D=np.diag(aff.sum(axis=1))
  # L = D - aff
  # # D_invdiv2=np.diag(np.sqrt(1/A_knn.sum(axis=1)))
  # # L=np.eye(A_knn.shape[0])-D_invdiv2@A_knn@D_invdiv2
  # L=sparse.csr_matrix(L)
  # eigenValues, eigenVectors=eigsh(L,k=3,which='SM')
  # idx = np.argsort(eigenValues)
  # Phi0 = eigenVectors[:,1:]

  # plot_2d_scatter(Phi0, title=f"Graph Laplacian VDM affinity: {N}, K={K}")

plt.show()
