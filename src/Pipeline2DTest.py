from cmath import pi
from utils.SingleImagePipeline import SingleImagePipeline
from utils.ImagesPipeline import ImagesPipeline
from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import numpy as np
import time

from utils.CoCoDataset import *
from utils.Plotting import *

# ########### Parameters #############
# data
RESOLUTION = 200
N = 512
DOUBLE_ANGLES = True
DOUBLE_PROJECTIONS = False

ADD_CIRCLE_PADDING = True

#graph
K = 9

#noise
SNR=25

# kernel
# epsilon = 0.5

# angle_generation = 'linear_spaced'
angle_generation = 'uniform'
use_wandb = True

n_imgs = 10

image_path = "src/data/val2017/"
files = os.listdir(image_path)
x = load_images_files(image_path, files, RESOLUTION,RESOLUTION, number=n_imgs, num_seed=5)

plot_imshow(x[0].reshape((RESOLUTION, RESOLUTION)))

names = np.array([f"image-{i}" for i in range(n_imgs) ])
pipeline = ImagesPipeline(x, names, RESOLUTION, add_circle_padding=ADD_CIRCLE_PADDING, verbose=True, noise_only=False)
pipeline.run(samples=N, angle_generation=angle_generation, double_angles=DOUBLE_ANGLES, snr=SNR, k=K, reconstruction_angles='linear_spaced', log_wandb=True)

# plot_imshow(x[1].reshape((RESOLUTION, RESOLUTION)))


# input = shepp_logan_phantom() # 400, 400

# for i in range(n_imgs):
#   t = time.time()
#   input = x[i].reshape((RESOLUTION, RESOLUTION))
#   pipeline = SingleImagePipeline(input, f"image-{i}", RESOLUTION, add_circle_padding=ADD_CIRCLE_PADDING, verbose=True, noise_only=False)
#   loss, loss_noisy =  pipeline.run(N, angle_generation=angle_generation, double_angles=DOUBLE_ANGLES, snr=SNR, k=K, reconstruction_angles='linear_spaced', log_wandb=True)

#   print("Loss: ", loss)  
#   print("Loss noisy: ", loss_noisy)  



# pipeline.plot_reconstruction_image(show=False)
# pipeline.plot_reconstruction_noisy_image(show=False)
# pipeline.plot_forward_image(show=False)
# pipeline.plot_forward_noisy_image(show=False)
# pipeline.plot_forward_graph_laplacian(show=False)
# pipeline.plot_forward_noisy_graph_laplacian(show=False)
# plt.show()

# for snr in range(15,30,3):
#   pipeline.log_wandb()
#   pipeline.reset()