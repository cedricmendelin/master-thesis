from  utils.ToyImageGenerator import draw_uniform_toyimages
from utils.Plotting import plot_imshow

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse, random_shapes

location = "data/toyimages_validation/64_shape_size/"
n = 5

# same logic as training images
for shape_size in range(8,17,1):
  images = draw_uniform_toyimages(64, shape_size, n)    
  for i in range(n):
    plt.gray()
    # plot_imshow(images[i])
    # plt.show()
    # plt.imsave(location + f"val_image_{i}_{shape_size}.png", images[i])


location = "data/toyimages_validation/64_ellipse/"
# some ellipses
for rot in range(30, 151,30):
  img = np.zeros((64, 64), dtype=np.uint8)
  rr, cc = ellipse(30, 30, 10, 15, rotation=np.deg2rad(rot))
  img[rr, cc] = 1
  # plt.imsave(location + f"val_ellipse_{rot}.png", img)

# some random shapes from ski-image:  
location = "data/toyimages_validation/64_random_shapes/"
resolution = 64
samples = 100
for i in range(samples):
  image, labels = random_shapes((64, 64), min_shapes=1, max_shapes=4, min_size=10, max_size=16, multichannel=False, intensity_range=(0,1))
  image[image == 255] = 0
  plt.imsave(location + f"val_random_{i}.png", image)




#plt.show()
