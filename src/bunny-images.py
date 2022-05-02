import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

loaded = np.load("src/data/original_images_bunny_1000_100.npz")
original_images = loaded['original_images']

print (original_images.shape)

n = 1000

for i in range(n):
  plt.imsave(f"src/bunnies/bunny_{i}.png", original_images[i,:,:], cmap = cm.gray)