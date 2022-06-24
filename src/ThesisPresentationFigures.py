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

import vedo
from vedo import dataurl, Mesh, volumeFromMesh, Points

torch.manual_seed(2022)
np.random.seed(2022)

font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)


def chapter_imaging_sinos(phantom = None):
  resolution = 400
  samples = 500
  snr = 0

  if phantom is None:
    phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # checkpoint = torch.load("models/unet_128.pt", map_location=device)
  # unet = UNet(nfilter=128).eval()
  # unet.load_state_dict(checkpoint['model_state_dict'])    

  radon, fbp, pad =  setup_forward_and_backward(resolution, samples)

  sino = radon(phantom).data[:, resolution // 2:resolution //2 + resolution]
  sino_noisy = add_noise_np(snr, sino)
  reconstruction = fbp(pad(torch.from_numpy(sino)))
  reconstruction_snr = fbp(pad(torch.from_numpy(sino_noisy)))

  # reconstruction_fbp_unet = unet(
  #   torch.from_numpy(reconstruction_snr.data).view(-1, 1, resolution, resolution)).view(resolution, resolution).cpu().detach().numpy()

  plot_imshow(phantom, title="Shepp Logan Phantom", colorbar=False)
  plot_imshow(sino, title="Sinogram", xlabel="s", ylabel='$\\theta$', colorbar=False)
  plot_imshow(sino_noisy, title=f"Sinogram with noise SNR: {snr} dB", xlabel="s", ylabel='$\\theta$', colorbar=False)
  plot_imshow(reconstruction, title="FBP clean sinogram", colorbar=False)
  plot_imshow(reconstruction_snr, title="FBP noisy sinogram", colorbar=False)
  # plot_imshow(reconstruction_fbp_unet, title="FBP + U-Net noisy sinogram", colorbar=False)

  print(find_SNR(torch.from_numpy(phantom), torch.from_numpy(reconstruction_snr.data)))
  # print(find_SNR(torch.from_numpy(phantom), reconstruction_fbp_unet))
  
  plt.show()

def chapter_graph_foundation_manifolds_different_k(phantom = None):
  resolution = 200
  samples = 500
  if phantom is None:
    phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

  radon, fbp, pad =  setup_forward_and_backward(resolution, samples)

  # clean graph
  sino = radon(phantom).data[:, resolution // 2:resolution //2 + resolution]
  #n_neighbors = 2
  plt.figure(figsize=(10,10))
  
  for k in range(2,11):
    eVec = get_embedding(sino, k, 2)
    plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")

  lgnd = plt.legend([f"k = {i}" for i in range(2,11)], loc='upper left')
  for handle in lgnd.legendHandles:
    handle.set_sizes([40.0])

  # plt.title("Manifold clean sinogram different k")
  plt.xticks([-0.08, -0.04, 0 , 0.04, 0.08])
  plt.yticks([-0.08, -0.04, 0 , 0.04, 0.08])

  # sino_noisy = add_noise_np(snr, sino)
  # eVec = get_embedding(sino_noisy, 3, 2)
  # plt.scatter(eVec[:, 0], eVec[:, 1])

  #plot_2d_scatter(embedding, title='Graph Laplacian Shepp-Logan Phantom Sinogram')
  plt.show()


def estimate_angles(graph_laplacian, degree=False):
  # arctan2 range [-pi, pi]
  angles = np.arctan2(graph_laplacian[:,0],graph_laplacian[:,1]) + np.pi
  # sort idc ascending, [0, 2pi]
  idx  = np.argsort(angles)

  if degree:
    return np.degrees(angles), idx, np.degrees(angles[idx])
  else:
    return angles, idx, angles[idx]

def chapter_graph_foundation_manifolds_clean(phantom = None):
  resolution = 200
  samples = 500
  if phantom is None:
    phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

  radon, _, _=  setup_forward_and_backward(resolution, samples)

  # clean graph
  sino = radon(phantom).data[:, resolution // 2:resolution //2 + resolution]
  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino, 2, 2)
  _,idx2,angles2 = estimate_angles(eVec)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, c=angles2,cmap='hsv')
  plt.xticks([-0.06, -0.03, 0 , 0.03, 0.06])
  plt.yticks([-0.06, -0.03, 0 , 0.03, 0.06])
  # plt.title("Manifold clean sinogram k = 2")
  plt.show()


def chapter_graph_foundation_manifolds_noisy(phantom = None):
  resolution = 200
  samples = 500
  if phantom is None:
    phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

  snr_1 = 20
  snr_2 = 0
  radon, fbp , pad =  setup_forward_and_backward(resolution, samples)

  # clean graph
  sino = radon(phantom).data[:, resolution // 2:resolution //2 + resolution]

  sino_noisy_1 = add_noise_np(snr_1, sino)
  sino_noisy_2 = add_noise_np(snr_2, sino)

  eVec_clean = get_embedding(sino, 6, 2)

  plt.figure(figsize=(10,10))
  eVec1 = get_embedding(sino_noisy_1, 6, 2)
  _,idx1,angles1 = estimate_angles(eVec1)
  plt.scatter(eVec1[:, 0], eVec1[:, 1], s=4, c=angles1,cmap='hsv')
  plt.xticks([-0.08, -0.04, 0 , 0.04, 0.08])
  plt.yticks([-0.08, -0.04, 0 , 0.04, 0.08])
  # plt.title(f"Manifold noisy sinogram k = 2, SNR={snr}dB")

  plt.figure(figsize=(10,10))
  eVec2 = get_embedding(sino_noisy_2, 8, 2)
  _,idx2,angles2 = estimate_angles(eVec2)
  plt.scatter(eVec2[:, 0], eVec2[:, 1], s=4, c=angles2,cmap='hsv')
  plt.xticks([-0.08, -0.04, 0 , 0.04, 0.08])
  plt.yticks([-0.1, -0.05, 0 , 0.03, 0.06])
  # plt.title(f"Manifold noisy sinogram k = 4, SNR={snr}dB")

  _,idx,angles = estimate_angles(eVec_clean)
  _, fbp_clean,_ =  setup_forward_and_backward(resolution, samples, angles)
  _, fbp1,_ =  setup_forward_and_backward(resolution, samples, angles1)

  # _, fbp2,_ =  setup_forward_and_backward(resolution, samples, angles2)

  reco_clean = fbp_clean(pad(torch.from_numpy(sino[idx])))
  reco_1 = fbp1(pad(torch.from_numpy(sino_noisy_1[idx1])))
  # reco_2 = fbp2(pad(torch.from_numpy(sino_noisy_2[idx2])))

  plot_imshow(reco_clean, colorbar=False, size=(10,10))
  plot_imshow(reco_1, colorbar=False, size=(10,10))
  # plot_imshow(reco_2, colorbar=False, size=(10,10))

  plot_imshow(fbp(pad(torch.from_numpy(sino))), colorbar=False, size=(10,10))
  plot_imshow(fbp(pad(torch.from_numpy(sino_noisy_1))), colorbar=False, size=(10,10))
  # plot_imshow(fbp2(pad(torch.from_numpy(sino_noisy_2))), colorbar=False, size=(10,10))


  # plt.figure(figsize=(10,10))
  # # plt.title("Manifold noisy sinogram different k")
  # for k in range(3,11):
  #   eVec = get_embedding(sino_noisy, k, 2)
  #   plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")

  # lgnd = plt.legend([f"k = {i}" for i in range(3,11)], loc='upper left')
  # for handle in lgnd.legendHandles:
  #   handle.set_sizes([40.0])

  # plt.xticks([-0.08, -0.04, 0 , 0.04, 0.08])
  # plt.yticks([-0.08, -0.04, 0 , 0.04, 0.08])

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
  samples = 200
  samples_2 = 500
  phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

  radon, _, _=  setup_forward_and_backward(resolution, samples)
  radon2, _, _=  setup_forward_and_backward(resolution, samples_2)

  # clean graph
  sino = radon(phantom).data[:, resolution // 2:resolution //2 + resolution]
  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino, 6, 2)
  _,_,angles = estimate_angles(eVec)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, c=angles,cmap='hsv')
  plt.xticks([-0.1, -0.05, 0 , 0.05, 0.1])
  plt.yticks([-0.1, -0.05, 0 , 0.05, 0.1])
  # plt.title("Manifold clean sinogram k = 6, 500 samples")
  
   # clean graph
  sino = radon2(phantom).data[:, resolution // 2:resolution //2 + resolution]
  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino, 6, 2)
  _,_,angles = estimate_angles(eVec)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, c=angles,cmap='hsv')
  plt.xticks([-0.06, -0.03, 0 , 0.03, 0.06])
  plt.yticks([-0.06, -0.03, 0 , 0.03, 0.06])
  # plt.title("Manifold clean sinogram k = 6, 700 samples")
  
  plt.show()

def chapter_graph_foundation_manifolds_clean_resolution_importance():
  resolution_2 = 200
  resolution = 400
  samples = 500
  k = 6
  phantom_original = shepp_logan_phantom()
  phantom = rescale(phantom_original, scale=resolution / phantom_original.shape[0], mode='reflect')
  phantom_2 = rescale(phantom_original, scale=resolution_2 / phantom_original.shape[0], mode='reflect')

  radon, _, _=  setup_forward_and_backward(resolution, samples)
  radon2, _, _ =  setup_forward_and_backward(resolution_2, samples)

  # clean graph
  sino = radon(phantom).data[:, resolution // 2:resolution //2 + resolution]
  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino, k, 2)
  _,_,angles = estimate_angles(eVec)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, c=angles,cmap='hsv')
  plt.xticks([-0.06, -0.03, 0 , 0.03, 0.06])
  plt.yticks([-0.06, -0.03, 0 , 0.03, 0.06])
  # plt.title(f"Manifold clean sinogram k = {k}, resolution={resolution}")
  
   # clean graph
  sino = radon2(phantom_2).data[:, resolution_2 // 2:resolution_2 //2 + resolution_2]
  plt.figure(figsize=(10,10))
  eVec = get_embedding(sino, k, 2)
  _,_,angles = estimate_angles(eVec)
  plt.scatter(eVec[:, 0], eVec[:, 1], s=4, c=angles,cmap='hsv')
  # plt.scatter(eVec[:, 0], eVec[:, 1], s=4, marker="o")
  plt.xticks([-0.06, -0.03, 0 , 0.03, 0.06])
  plt.yticks([-0.06, -0.03, 0 , 0.03, 0.06])
  # plt.title(f"Manifold clean sinogram k = {k}, resolution={resolution_2}")
  
  plt.show()


#chapter_imaging_sinos()
#chapter_graph_foundation_cricle_manifold()
#chapter_graph_foundation_sphere_manifold()

# chapter_graph_foundation_manifolds_different_k()
# chapter_graph_foundation_manifolds_clean()
#chapter_graph_foundation_manifolds_noisy()
# chapter_graph_foundation_manifolds_noisy()
#chapter_graph_foundation_manifolds_clean_samples_importance()
#chapter_graph_foundation_manifolds_clean_resolution_importance()
#chapter_results_small_overall_compoents()
#chapter_results_large_overall_compoents()
#test_wandb_api()
#wandb_export_project("LoDoPaB-Large-knn-2")  
#wandb_rename("LoDoPaB-Large-knn-2", "1y8gw6kv", "conv_gat_unet_train_snr--15_epochs20_3")


def bunny():
  mesh = Mesh(dataurl + "bunny.obj").c("gold")

  vol = Points.tovolume(mesh.points(), radius=0.5)
  # vol = volumeFromMesh(mesh, spacing=(0.01,0.01,0.01))

  print(vol.shape())



def observation_bunny_noisy():
  import PIL
  image = PIL.Image.open('data/bunnies/bunny_471.png')
  # convert image to numpy array
  data = np.asarray(image)[:,:,0]
  print(data.shape)
  # plt.style.use('dark_background')
  data = data / 255

  print(np.min(data))
  print(np.max(data))

  noisy_data = add_noise_np(-5, data)

  plot_imshow(data, colorbar=False)
  plot_imshow(noisy_data, colorbar=False)
  plt.show()


# observation_bunny_noisy()

def my_foot_observation():
  import PIL
  image = PIL.Image.open('src/foot_cm.jpg')
  data = np.asarray(image)[:,:,0]
  lin = np.linspace(-1,1, data.shape[0])
  XX, YY = np.meshgrid(lin,lin)
  circle = ((XX**2+YY**2)<=1)*1.
  data = data * circle
  # chapter_imaging_sinos(data)
  chapter_graph_foundation_manifolds_noisy(data)

my_foot_observation()
