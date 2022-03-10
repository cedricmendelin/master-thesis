import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import default_rng

from utils.Graph import *
from utils.Plotting import *

np.set_printoptions(precision=4)

# Parameters
N = 100
K = 4
debug_plot = True
debug_print = False

############### Sampling points on circle ###########################

step = 2*np.pi / N
angle_in_radian = np.linspace(0, 2*np.pi - step,N)

rng = default_rng(1)
# noise = rng.standard_normal(N)
# noisy_angle_in_radian = angle_in_radian + noise

point_on_cricle = np.array([np.cos(angle_in_radian), np.sin(angle_in_radian)]).T
noisy_points_on_circle = point_on_cricle + np.array([rng.standard_normal(N), rng.standard_normal(N)]).T

if debug_plot:
  plt.figure()
  plt.scatter(point_on_cricle[:,0], point_on_cricle[:,1], s=20)
  plt.xticks([-1, -0.5,0,0.5,1])
  plt.yticks([-1, -0.5,0,0.5,1])
  plt.title("clean points")

  plt.figure()
  plt.scatter(noisy_points_on_circle[:,0], noisy_points_on_circle[:,1], s=20)
  plt.xticks([-1, -0.5,0,0.5,1])
  plt.yticks([-1, -0.5,0,0.5,1])
  plt.title("noisy points")

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

noisy_distances = np.array([great_circle_distance(noisy_points_on_circle[i,0], noisy_points_on_circle[i,1], noisy_points_on_circle[j,0], noisy_points_on_circle[j,1], i) for i in range(N) for j in range(N)]).reshape((N,N))
noisy_distances /= noisy_distances.max()

if debug_print:
  print(f"Distances: {distances}")
  print(f"Distances: {distances.max()}")
  print(f"Distances: {distances.min()}")
  print(f"Distances: {distances.mean()}")


print("Distances done!")

############### Generating Knn graph #########################

graph, classes = generate_knn_from_distances(distances, K, ordering='asc', ignoreFirst=True)
noisy_graph, noisy_classes = generate_knn_from_distances(noisy_distances, K, ordering='asc', ignoreFirst=True)

if debug_plot:
  graph_laplacian = calc_graph_laplacian_nx(graph, embedDim=2)
  plot_2d_scatter(graph_laplacian, title=f"Graph Laplacian input graph")

  graph_laplacian_noisy = calc_graph_laplacian_nx(noisy_graph, embedDim=2)
  plot_2d_scatter(graph_laplacian_noisy, title=f"Graph Laplacian noisy input graph")

if debug_print:
  print("Graph: ", graph)
  print("Neighbours", classes)

print("Input Graph done!")

plt.show()