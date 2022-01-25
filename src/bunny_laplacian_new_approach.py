from utils.Graph import *
from utils.Data import *
from utils.Plotting import *
from utils.DiffusionMaps import *
import matplotlib.pyplot as plt
import networkx as nx

N = 1000
image_res = 100
snr = 10
K=10

original_images, uniform_3d_angles, noisy_images, noise = create_or_load_bunny_immages(N, image_res, snr, save=True)

classes, reflections, rotations, correlations = create_or_load_knn_rotation_invariant(name='bunny', N=N, image_res=image_res, images=original_images, K=K)
classes_noisy, reflections_noisy, rotations_noisy, correlations_noisy = create_or_load_knn_rotation_invariant(name='bunny_noisy', N=N, image_res=image_res, images=noisy_images, K=K, snr=snr)

#dist_knn, idx_best_img_knn, angle_est_knn = rotation_invariant_knn(original_images, K=K)
#dist_knn_noisy, idx_best_img_knn_noisy, angle_est_knn_noisy = rotation_invariant_knn(noisy_images, K=K)


#Laplacian
#A = create_adj_mat(classes)
A = create_adj_mat(classes, reflections)

embedding = calc_graph_laplacian(A, numberOfEvecs=3)

#print(embedding.shape)

plot_3dscatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],  (10,10))


A_noisy = create_adj_mat(classes_noisy, reflections_noisy)

embedding_noisy = calc_graph_laplacian(A_noisy, numberOfEvecs=3)

print(embedding_noisy.shape)

plot_3dscatter(embedding_noisy[:, 0], embedding_noisy[:, 1], embedding_noisy[:, 2],  (10,10))


diff = embedding - embedding_noisy

#assert np.all(diff != 0) , "embeddings are same"




input("Enter to terminate")