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

class SingleImagePipeline():
  def __init__(
    self,
    image,
    name,
    resolution,
    add_circle_padding=True,
    noise_only=True,
    verbose=True):

    self.N = None
    self.DOUBLE_ANGLES = False
    self.DOUBLE_PROJECTIONS = False
    self.RESOLUTION = resolution
    self.verbose = verbose
    self.NOISE_ONLY = noise_only
    self.name = name
    self.reset()

    if self.verbose:
      t = time.time()

    if image.shape[0] == resolution and image.shape[1] == resolution:
      self.input_image = image
    else:
      scaleX = resolution / image.shape[0]
      scaleY = resolution / image.shape[1]
      self.input_image =  rescale(image, scale=(scaleX, scaleY), mode='reflect', multichannel=False)

    if add_circle_padding:
      r = np.ceil(np.sqrt(2 * (self.RESOLUTION ** 2))) 
      padding = int(np.ceil((r - self.RESOLUTION) / 2))
      self.padding = padding
      p = (padding, padding)
      self.input_image = np.pad(self.input_image, [p,p], mode='constant', constant_values=0)
      self.RESOLUTION += 2 * padding

    if self.verbose:
      self.time_dict["init"] = time.time()-t

  def _get_forward_image(self):
    return self.forward_image[np.argsort(self.input_angles_degrees)]

  def _get_forward_image_noisy(self):
    return self.forward_noisy_image[np.argsort(self.input_angles_degrees)]

  def _plot_image(self, image, title='', c_map=color_map, show=True, aspect=None):
    assert image is not None, "Cannot plot image, it is set to None."
    plot_imshow(image, title=title, c_map=c_map, show=show)

  def plot_original_image(self, show=True):
    self._plot_image(self.input_image, title='Original image', c_map=color_map, show=show)

  def plot_forward_image(self, show=True):
    self._plot_image(
      self._get_forward_image(),
      title='Sinogram original image', 
      c_map=reversed_color_map, 
      show=show, 
      aspect='auto')

  def plot_forward_noisy_image(self, show=True):
    self._plot_image(
      self._get_forward_image_noisy(), 
      title='Sinogram noisy image', 
      c_map=reversed_color_map, 
      show=show, 
      aspect='auto')
  
  def plot_reconstruction_noisy_image(self, show=True):
    self._plot_image(
      self.reconstructed_noisy_image, 
      title='Reconstructed noisy image', 
      c_map=color_map, 
      show=show)

  def plot_reconstruction_image(self, show=True):
    self._plot_image(
      self.reconstructed_image, 
      title='Reconstructed clean noisy image', 
      c_map=color_map, 
      show=show)

  def _plot_2d_scatter(self, data, title, show=True):
    assert data is not None, "Cannot plot data, it is set to None."
    plot_2d_scatter(data, title=title, show=show)

  def plot_forward_graph_laplacian(self, show=True):
    self._plot_2d_scatter(self.forward_graph_laplacian, f"Graph Laplacian sinogram graph K = {self.K}", show=show)
    
  def plot_forward_noisy_graph_laplacian(self, show=True):
    self._plot_2d_scatter(self.forward_noisy_graph_laplacian, f"Graph Laplacian noisy sinogram graph K = {self.K}", show=show)

  def setup_angles(self, samples=1024, angle_generation = 'uniform', double_angles=True):
    assert angle_generation == 'uniform' or angle_generation == 'linear_spaced', "Angle Generation method unknown"
    if self.verbose:
      t = time.time()

    self.ANGLE_GENERATION = angle_generation
    self.DOUBLE_ANGLES = double_angles
    self.number_of_samples = samples
    self.N = samples

    if angle_generation == 'linear_spaced':
      self.input_angles = np.linspace(0, 2 * np.pi, self.N)

    if angle_generation == 'uniform':
      rng = default_rng(SEED)
      self.input_angles = rng.uniform(0, 2 * np.pi, self.N)

    if double_angles:
      self.input_angles = np.concatenate((self.input_angles, np.mod(self.input_angles + np.pi , 2 * np.pi)))
      self.number_of_samples = self.number_of_samples * 2
    
    self.input_angles_degrees = np.degrees(self.input_angles)
    if self.verbose:
      self.time_dict["init"] = time.time()-t

  def forward(self, snr=20, distances=True, knn=True, K=8, double_projections=False):
    assert not double_projections or not self.DOUBLE_ANGLES , "Cannot double projections and angles"
    
    self.DOUBLE_PROJECTIONS = double_projections
    self.SNR = snr

    self._execute_and_log_time(lambda: self._calc_forward(), "forward")

    if distances:
      self._execute_and_log_time(lambda: self._calculate_forward_l2_distances(), "calc_distances")
    if knn:
      self._execute_and_log_time(lambda: self._calculate_input_knn(K), "calc_graphs" )

  def reconstruct_with_graph_laplacian_angles(self, reconstruction_angles='linear_spaced'):
    assert reconstruction_angles == 'linear_spaced', "Angle Generation method unknown"
    self.reconstruction_angles = reconstruction_angles

    self._execute_and_log_time(lambda: self._reconstruct_with_graph_laplacian_angles(), "reconstruction")
   
  def loss_clean(self):
    assert self.input_image is not None , "Cannot compute loss as input image is not available"
    assert self.reconstructed_image is not None , "Cannot compute loss as reconstruction image is not available"

    return np.linalg.norm(self.input_image - self.reconstructed_image)
    
  def loss_noisy(self):
    assert self.input_image is not None , "Cannot compute loss as input image is not available"
    assert self.reconstructed_noisy_image is not None , "Cannot compute loss as reconstruction image is not available"

    return np.linalg.norm(self.input_image - self.reconstructed_noisy_image)

  def reset(self):
    self.number_of_samples = self.N
    if self.DOUBLE_ANGLES:
      self.number_of_samples = int(self.number_of_samples * 2)

    if self.DOUBLE_PROJECTIONS:
      self.input_angles = self.input_angles[0:self.number_of_samples]
      self.input_angles_degrees = np.degrees(self.input_angles)

    self.DOUBLE_PROJECTIONS = False
    self.SNR = None
    self.forward_image = None
    self.forward_noisy_image = None
    self.distances = None
    self.noisy_distances = None
    self.K = None
    self.input_graph, self.input_graph_classes = None, None
    self.noisy_graph, self.noisy_graph_classes = None, None
    self.reconstruction_angles = None
    self.forward_graph_laplacian = None
    self.forward_noisy_graph_laplacian = None
    self.reconstructed_image = None
    
    self.reconstructed_noisy_image = None
    self.time_dict = {}

  def run(self, samples=1024, angle_generation='uniform', double_angles=True,snr=25, k=9, reconstruction_angles='linear_spaced', log_wandb=False, reset=False):
    t = time.time()
    self.setup_angles(samples, angle_generation, double_angles)
    self.forward(snr, True, True, k, False)
    self.reconstruct_with_graph_laplacian_angles(reconstruction_angles)
    
    if log_wandb:
      self.log_wandb(execution_time=time.time() - t)

    if reset:
      self.reset()

    if self.NOISE_ONLY :
      return self.loss_noisy()

    return self.loss_clean(), self.loss_noisy()

  def _calc_forward(self):
    Rf = radon(self.input_image, theta=self.input_angles_degrees, circle=True)
    self.forward_image = Rf.T

    if self.DOUBLE_PROJECTIONS:
      # double projections
      self.forward_image = np.concatenate((self.forward_image, np.flip(self.forward_image, 1)))
      
      # update angles and other parameters
      self.input_angles = np.concatenate((self.input_angles, np.mod(self.input_angles + np.pi , 2 * np.pi)))
      self.number_of_samples = self.number_of_samples * 2
      self.input_angles_degrees = np.degrees(self.input_angles)

    self.forward_noisy_image = self._add_noise(self.SNR, self.forward_image)

  def _add_noise(self, SNR, sinogram):
    self.SNR = SNR
    sinogram=np.array(sinogram)
    VARIANCE=10**(-SNR/10)*(np.std(sinogram)**2)
    noise = np.random.randn(sinogram.shape[0],sinogram.shape[1])*np.sqrt(VARIANCE)
    return sinogram + noise

  def _calculate_forward_l2_distances(self):
    if not self.NOISE_ONLY:
      self.distances = self._calculate_l2_distances(self.forward_image)
    self.noisy_distances = self._calculate_l2_distances(self.forward_noisy_image)

  def _calculate_input_knn(self, K):
    self.K = K
    if not self.NOISE_ONLY:
      self.input_graph, self.input_graph_classes = generate_knn_from_distances(self.distances, K=K, ordering='asc', ignoreFirst=True)
    self.noisy_graph, self.noisy_graph_classes = generate_knn_from_distances(self.noisy_distances, K=K, ordering='asc', ignoreFirst=True)

  def _calculate_l2_distances(self, sinogram):
    N = sinogram.shape[0]
    dist = np.array([ np.linalg.norm(sinogram[i] - sinogram[j]) for i in range(N) for j in range(N)]).reshape((N,N))
    dist /= dist.max()
    return dist

  def _estimate_angles(self, graph_laplacian, degree=False):
    # arctan2 range [-pi, pi]
    angles = np.arctan2(graph_laplacian[:,0],graph_laplacian[:,1]) + np.pi
    # sort idc ascending, [0, 2pi]
    idx  = np.argsort(angles)

    if degree:
      return np.degrees(angles), idx, np.degrees(angles[idx])
    else:
      return angles, idx, angles[idx]

  def _reconstruct_image(self, image, idx, angles):
    sinogram_sorted = image[idx]
    return iradon(sinogram_sorted.T, theta=angles, circle=True)

  def _reconstruct_with_graph_laplacian_angles(self):
    angles = np.linspace(0, 360, self.number_of_samples)

    # clean case:
    if not self.NOISE_ONLY:
      self.forward_graph_laplacian = calc_graph_laplacian(self.input_graph, embedDim=2)
      _, clean_estimated_angles_indices, _  = self._estimate_angles(self.forward_graph_laplacian, degree=True)
      self.reconstructed_image = self._reconstruct_image(self.forward_image, clean_estimated_angles_indices, angles )

    # noisy case:
    self.forward_noisy_graph_laplacian = calc_graph_laplacian(self.noisy_graph, embedDim=2)
    _, noisy_estimated_angles_indices, _  = self._estimate_angles(self.forward_noisy_graph_laplacian, degree=True)
    self.reconstructed_noisy_image = self._reconstruct_image(self.forward_noisy_image, noisy_estimated_angles_indices, angles )

  def _execute_and_log_time(self, action, name):
      if self.verbose:
        t = time.time()

      action()
      
      if self.verbose:
        self.time_dict[name] = time.time()-t

  def log_wandb(self, entity="cedric-mendelin", project_name="2d-pipeline", execution_time=None):
    assert self.number_of_samples != None and self.SNR != None and self.K != None and self.ANGLE_GENERATION != None, "Pipeline not finished yet, cannot log."
    
    import wandb
  
    config = {
      "samples": self.number_of_samples,
      "resolution": self.RESOLUTION,
      "noise_SNR": self.SNR,
      "k-nn": self.K,
      "double_angles" : self.DOUBLE_ANGLES,
      "doube_projections" : self.DOUBLE_PROJECTIONS,
      "angle_generation" : self.ANGLE_GENERATION, 
      "name" : self.name
    }
    
    wandb.init(project=project_name, entity=entity, config=config, reinit=True)

    wandb.log({"input_image": wandb.Image(self.input_image)})
    wandb.log({"forward_image": wandb.Image(self._get_forward_image())})
    wandb.log({"forward_noisy_image": wandb.Image(self._get_forward_image_noisy())})

    wandb.log({"noisy GL" : wandb.plot.scatter(wandb.Table(data=self.forward_noisy_graph_laplacian, columns = ["x", "y"]),"x","y", title=f"GL noisy sinogram")})
    wandb.log({"reconstruction_noisy": wandb.Image(self.reconstructed_noisy_image)})
    if not self.NOISE_ONLY:
      wandb.log({"clean GL" : wandb.plot.scatter(wandb.Table(data=self.forward_graph_laplacian, columns = ["x", "y"]),"x","y", title=f"GL sinogram")})
      wandb.log({"reconstruction_clean": wandb.Image(self.reconstructed_image)})
    

    if execution_time != None:
      wandb.log({"execution_time" : execution_time})
    
    if self.verbose:
      wandb.log(self.time_dict)

    wandb.run.name = f"{self.name}-{self.RESOLUTION}-{self.number_of_samples}-{self.SNR}-{self.K}-{self.DOUBLE_ANGLES}"

    wandb.finish()


  