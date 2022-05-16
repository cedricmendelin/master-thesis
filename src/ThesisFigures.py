import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np

from utils.Plotting import *
from utils.ODLHelperCustom import setup_forward_and_backward
from utils.SNRHelper import *

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale


def chapter_imaging_sinos():
  resolution = 400
  samples = 500
  snr = 10
  phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')



  radon, fbp =  setup_forward_and_backward(resolution, samples)

  sino = radon(phantom)
  sino_noisy = add_noise_np(snr, sino)
  reconstruction = fbp(sino)
  reconstruction_snr = fbp(sino_noisy)


  plot_imshow(phantom, title="Shepp Logan Phantom", colorbar=False)
  plot_imshow(sino, title="Sinogram", xlabel="s", ylabel='$\\theta$', colorbar=False)
  plot_imshow(sino_noisy, title=f"Sinogram with noise SNR: {snr}", xlabel="s", ylabel='$\\theta$', colorbar=False)
  plot_imshow(reconstruction, title="FBP clean sinogram", colorbar=False)
  plot_imshow(reconstruction_snr, title="FBP noisy sinogram", colorbar=False)

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

  points = fibonacci_sphere(400)

  xx = points[:,0]
  yy = points[:,1]
  zz = points[:,2]


  #Set colours and render
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.plot_surface(
      x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

  ax.scatter(xx,yy,zz,color="k",s=20)

  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
  ax.set_zlim([-1,1])
  #ax.set_aspect("equal")
  ax.set_xticks([-1, -0.5,0,0.5,1])
  ax.set_yticks([-1, -0.5,0,0.5,1])
  ax.set_zticks([-1, -0.5,0,0.5,1])
  plt.tight_layout()
  plt.show()

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
#chapter_imaging_sinos()
#chapter_graph_foundation_cricle_manifold()
chapter_graph_foundation_sphere_manifold()
