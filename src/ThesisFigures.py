import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np

from utils.Plotting import *
from utils.ODLHelperCustom import setup_forward_and_backward
from utils.SNRHelper import *
from utils.Graph import *

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from scipy.spatial import distance_matrix
from sklearn.neighbors import kneighbors_graph

from utils.UNetModel import UNet

np.random.seed(2022)


def chapter_imaging_sinos():
  resolution = 400
  samples = 500
  snr = 10
  phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  checkpoint = torch.load("models/unet.pt", map_location=device)
  unet = UNet(nfilter=128).eval()
  unet.load_state_dict(checkpoint['model_state_dict'])    

  radon, fbp =  setup_forward_and_backward(resolution, samples)

  sino = radon(phantom)
  sino_noisy = add_noise_np(snr, sino)
  reconstruction = fbp(sino)
  reconstruction_snr = fbp(sino_noisy)

  reconstruction_fbp_unet = unet(
    torch.from_numpy(reconstruction_snr.data).view(-1, 1, resolution, resolution)).view(resolution, resolution).cpu().detach().numpy()

  plot_imshow(phantom, title="Shepp Logan Phantom", colorbar=False)
  plot_imshow(sino, title="Sinogram", xlabel="s", ylabel='$\\theta$', colorbar=False)
  plot_imshow(sino_noisy, title=f"Sinogram with noise SNR: {snr} dB", xlabel="s", ylabel='$\\theta$', colorbar=False)
  plot_imshow(reconstruction, title="FBP clean sinogram", colorbar=False)
  plot_imshow(reconstruction_snr, title="FBP noisy sinogram", colorbar=False)
  plot_imshow(reconstruction_fbp_unet, title="FBP + U-Net noisy sinogram", colorbar=False)

  plt.show()

def chapter_graph_foundation_cricle_manifold():
  from numpy.random import default_rng
  rng = default_rng()
  vals = rng.uniform(0, 2* np.pi, 200)

  plot_2dscatter(np.cos(vals), np.sin(vals), title="Uniform sampled circle manifold", figsize=(5,5))
  plt.show()

def chapter_graph_foundation_sphere_manifold():
  from mpl_toolkits.mplot3d import Axes3D
  import numpy as np
  from numpy.random import default_rng

  rng = default_rng()

  # Create a sphere
  r = 1
  pi = np.pi
  cos = np.cos
  sin = np.sin
  phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]

  x = r*sin(phi)*cos(theta)
  y = r*sin(phi)*sin(theta)
  z = r*cos(phi)

  #Import data
  #data = rng.uniform(0, 500, 600)
  #theta, phi, r = np.hsplit(data, 3) 
  #theta = theta * pi / 180.0
  #phi = phi * pi / 180.0
  rng = default_rng()
  phi = rng.uniform(0, 2* np.pi, 400)
  theta = rng.uniform(0, 2* np.pi, 400)


  #xx = sin(phi)*cos(theta)
  #yy = sin(phi)*sin(theta)
  #zz = cos(phi)

  points = sampling_sphere(800)

  xx = points[:,0]
  yy = points[:,1]
  zz = points[:,2]


  #Set colours and render
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, projection='3d')

  ax.plot_surface(
      x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

  ax.scatter(xx,yy,zz,color="gray",s=20)

  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
  ax.set_zlim([-1,1])
  #ax.set_aspect("equal")
  ax.set_xticks([-1, -0.5,0,0.5,1])
  ax.set_yticks([-1, -0.5,0,0.5,1])
  ax.set_zticks([-1, -0.5,0,0.5,1])
  plt.tight_layout()
  plt.show()

def chapter_graph_foundation_manifolds_different_k():
  resolution = 200
  samples = 500
  phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

  radon, fbp =  setup_forward_and_backward(resolution, samples)

  # clean graph
  sino = radon(phantom)
  #n_neighbors = 2
  plt.figure(figsize=(10,10))
  
  for k in range(2,11):
    eVec = get_embedding(sino, k, 2)
    plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")

  plt.legend([f"k-nn : {i}" for i in range(2,11)], loc='upper left')
  plt.title("Manifold clean sinogram different k")

  # sino_noisy = add_noise_np(snr, sino)
  # eVec = get_embedding(sino_noisy, 3, 2)
  # plt.scatter(eVec[:, 0], eVec[:, 1])

  #plot_2d_scatter(embedding, title='Graph Laplacian Shepp-Logan Phantom Sinogram')
  plt.show()


def chapter_graph_foundation_manifolds_clean():
  resolution = 200
  samples = 500
  phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

  radon, _ =  setup_forward_and_backward(resolution, samples)

  # clean graph
  sino = radon(phantom)
  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino, 2, 2)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")
  plt.title("Manifold clean sinogram k = 2")
  plt.show()

def chapter_graph_foundation_manifolds_noisy():
  resolution = 200
  samples = 500
  phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

  snr = 20
  radon, _ =  setup_forward_and_backward(resolution, samples)

  # clean graph
  sino = radon(phantom)

  sino_noisy = add_noise_np(snr, sino)

  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino_noisy, 2, 2)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")
  plt.title(f"Manifold noisy sinogram k = 2, SNR={snr}dB")

  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino_noisy, 4, 2)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")
  plt.title(f"Manifold noisy sinogram k = 4, SNR={snr}dB")

  plt.figure(figsize=(10,10))
  plt.title("Manifold noisy sinogram different k")
  for k in range(3,11):
    eVec = get_embedding(sino_noisy, k, 2)
    plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")

  plt.legend([f"k-nn : {i}" for i in range(3,11)], loc='upper left')

  plt.show()

def get_embedding(sino, knn, embed_dim):
    A_knn = kneighbors_graph(sino, n_neighbors=knn)
    A_knn = A_knn.toarray()
    #Compute the graph laplacian
    A_knn = 0.5*(A_knn + A_knn.T)
    L = np.diag(A_knn.sum(axis=1)) - A_knn
  
    #Extract the second and third smallest eigenvector of the Laplace matrix.
    eigenValues, eigenVectors = np.linalg.eigh(L)
    idx = np.argsort(eigenValues)
    return eigenVectors[:, idx[1 : embed_dim + 1]]

def fibonacci_sphere(samples=1000):

    points = np.zeros((samples, 3))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i] = np.array([x,y,z])

    return points

def sampling_sphere(samples):
    th = np.random.random(samples) * np.pi * 2
    x = np.random.random(samples) * 2 - 1
    out = np.array([np.cos(th) * np.sqrt(1 - x**2), np.sin(th) * np.sqrt(1 - x**2),x]).T
    return out


def chapter_graph_foundation_manifolds_clean_samples_importance():
  resolution = 200
  samples = 500
  samples_2 = 700
  phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

  radon, _ =  setup_forward_and_backward(resolution, samples)
  radon2, _ =  setup_forward_and_backward(resolution, samples_2)

  # clean graph
  sino = radon(phantom)
  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino, 6, 2)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")
  plt.title("Manifold clean sinogram k = 6, 500 samples")
  
   # clean graph
  sino = radon2(phantom)
  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino, 6, 2)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")
  plt.title("Manifold clean sinogram k = 6, 700 samples")
  
  plt.show()

def chapter_graph_foundation_manifolds_clean_resolution_importance():
  resolution_2 = 200
  resolution = 400
  samples = 600
  k = 5
  phantom_original = shepp_logan_phantom()
  phantom = rescale(phantom_original, scale=resolution / phantom_original.shape[0], mode='reflect')
  phantom_2 = rescale(phantom_original, scale=resolution_2 / phantom_original.shape[0], mode='reflect')

  radon, _ =  setup_forward_and_backward(resolution, samples)
  radon2, _ =  setup_forward_and_backward(resolution_2, samples)

  # clean graph
  sino = radon(phantom)
  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino, k, 2)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")
  plt.title(f"Manifold clean sinogram k = {k}, resolution=400")
  
   # clean graph
  sino = radon2(phantom_2)
  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino, k, 2)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")
  plt.title(f"Manifold clean sinogram k = {k}, resolution=200")
  
  plt.show()


#chapter_imaging_sinos()
#chapter_graph_foundation_cricle_manifold()
#chapter_graph_foundation_sphere_manifold()
#chapter_graph_foundation_manifolds_different_k()
#chapter_graph_foundation_manifolds_clean()
#chapter_graph_foundation_manifolds_noisy()
#chapter_graph_foundation_manifolds_clean_samples_importance
#chapter_graph_foundation_manifolds_clean_resolution_importance()
chapter_imaging_sinos()