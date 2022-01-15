from utils.Graph import *
from utils.Data import *
from utils.Plotting import *
from utils.DiffusionMaps import *
import matplotlib.pyplot as plt
import networkx as nx

N = 500
image_res = 100
snr = 10
K=8

original_images, uniform_3d_angles, noisy_images, noise = create_or_load_bunny_immages(N, image_res, snr, save=True)
classes, reflections, rotations, correlations = create_or_load_knn_rotation_invariant(name='bunny', N=N, image_res=image_res, images=original_images, K=K)


#Laplacian
#A = create_adj_mat(classes)
A = create_adj_mat(classes, reflections)

embedding = calc_graph_laplacian_from_knn_indices(classes, numberOfEvecs=3)

print(embedding.shape)

plot_3dscatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],  (10,10))


# Diffusion Maps
P = diffusion_map(X=A, alpha=1)

D, psi = diffusion_distance(P, n_eign=3,t=1)

coor1=(psi[:, 0]-psi[:, 0].mean())/psi[:, 0].std()
coor2=(psi[:, 1]-psi[:, 1].mean())/psi[:, 1].std()
coor3=(psi[:, 2]-psi[:, 2].mean())/psi[:, 2].std()


print(psi.shape)

print(D.shape)


plot_3dscatter(psi[:,0], psi[:,1], psi[:,2], (10,10))


input("Enter to terminate")