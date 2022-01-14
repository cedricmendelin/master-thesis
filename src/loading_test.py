import numpy as np
import utils.Plotting as plot

#np.savez_compressed('aspire_knn_original_500_10.np', dist, ind, angles)
#np.savez_compressed('original_images_500_100.np', original_images, uniform_3d_angles)

loaded = np.load('aspire_knn_original_500_10.np.npz')
print(loaded.files)

loaded2 = np.load('original_images_500_100.np.npz')

plot.plot_imshow(loaded2['arr_0'][1])

input("Enter to terminate")