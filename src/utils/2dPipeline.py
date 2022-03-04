from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

import time

from utils.Plotting import *
from utils.Graph import *
from utils.Data import find_sigma_noise

import math

SEED = 2022
color_map = plt.cm.get_cmap('gray')
reversed_color_map = color_map.reversed()

class Pipeline2D():
  def __init__(
    self,
    image,
    resolution,
    samples):

    self.resolution = resolution
    self.N = samples

    if image.shape[0] == resolution and image.shape[1] == resolution:
      self.input_image = image
    else:
      scaleX = resolution / image.shape[0]
      scaleY = resolution / image.shape[1]
      self.input_image =  rescale(image, scale=(scaleX, scaleY), mode='reflect', multichannel=False)



  def plot_original_image(self):
    plot_imshow(self.input_image, title='Original image', c_map=color_map)

  def plot_forward_image(self):
    plot_imshow(self.forward_image[np.argsort(self.input_angles_degrees)], title='Sinogram original image', aspect='auto', c_map=reversed_color_map)

  def plot_forward_noisy_image(self):
    plot_imshow(self.forward_noisy_image[np.argsort(self.input_angles_degrees)], title='Sinogram noisy image', aspect='auto', c_map=reversed_color_map)
  
  def setup_angles(self, angle_generation = 'uniform', double_angles=True):
    if angle_generation == 'linear_spaced':
      self.input_angles = np.linspace(0, 2 * np.pi, self.N)

    if angle_generation == 'uniform':
      rng = default_rng(SEED)
      self.input_angles = rng.uniform(0, 2 * np.pi, self.N)

    if double_angles:
      self.input_angles = np.concatenate((self.input_angles, np.mod(self.input_angles + np.pi , 2 * np.pi)))
      self.N = self.N * 2
    else:
      self.N = self.N

    self.input_angles_degrees = np.degree(self.input_angles)

  def forward(self, snr=20, distances=True, knn=True, K=8):
    Rf = radon(self.input_image, theta=self.input_angles_degrees, circle=True)
    self.forward_image = Rf.T
    self.forward_noisy_image = self.add_noise(snr, self.forward_image)

    if distances:
      self.calculate_forward_l2_distances()

    if knn:
      self.calculate_input_knn(K)

  def add_noise(SNR,sinogram):
    sinogram=np.array(sinogram)
    VARIANCE=10**(-SNR/10)*(np.std(sinogram)**2)
    noise = np.random.randn(sinogram.shape[0],sinogram.shape[1])*np.sqrt(VARIANCE)
    return sinogram + noise

  def calculate_forward_l2_distances(self):
    self.distances = self.calculate_l2_distances(self.forward_image)
    self.noisy_distances = self.calculate_l2_distances(self.forward_noisy_image)

  def calculate_input_knn(self, K):
    self.input_graph, self.input_graph_classes = generate_knn_from_distances(self.forward_image, K=K, ordering='asc', ignoreFirst=True)
    self.noisy_graph, self.noisy_graph_classes = generate_knn_from_distances(self.forward_noisy_image, K=K, ordering='asc', ignoreFirst=True)


  def calculate_l2_distances(self, sinogram):
    N = sinogram.shape[0]
    dist = np.array([ np.linalg.norm(sinogram[i] - sinogram[j]) for i in range(N) for j in range(N)]).reshape((N,N))
    dist /= dist.max()
    return dist

  def estimate_angles(self):
    