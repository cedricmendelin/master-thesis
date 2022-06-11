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

torch.manual_seed(2022)
np.random.seed(2022)

font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)


def chapter_imaging_sinos():
  resolution = 400
  samples = 500
  snr = 0
  phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  checkpoint = torch.load("models/unet_128.pt", map_location=device)
  unet = UNet(nfilter=128).eval()
  unet.load_state_dict(checkpoint['model_state_dict'])    

  radon, fbp, pad =  setup_forward_and_backward(resolution, samples)

  sino = radon(phantom).data[:, resolution // 2:resolution //2 + resolution]
  sino_noisy = add_noise_np(snr, sino)
  reconstruction = fbp(pad(torch.from_numpy(sino)))
  reconstruction_snr = fbp(pad(torch.from_numpy(sino_noisy)))

  reconstruction_fbp_unet = unet(
    torch.from_numpy(reconstruction_snr.data).view(-1, 1, resolution, resolution)).view(resolution, resolution).cpu().detach().numpy()

  plot_imshow(phantom, title="Shepp Logan Phantom", colorbar=False)
  plot_imshow(sino, title="Sinogram", xlabel="s", ylabel='$\\theta$', colorbar=False)
  plot_imshow(sino_noisy, title=f"Sinogram with noise SNR: {snr} dB", xlabel="s", ylabel='$\\theta$', colorbar=False)
  plot_imshow(reconstruction, title="FBP clean sinogram", colorbar=False)
  plot_imshow(reconstruction_snr, title="FBP noisy sinogram", colorbar=False)
  plot_imshow(reconstruction_fbp_unet, title="FBP + U-Net noisy sinogram", colorbar=False)

  print(find_SNR(torch.from_numpy(phantom), torch.from_numpy(reconstruction_snr.data)))
  print(find_SNR(torch.from_numpy(phantom), reconstruction_fbp_unet))
  
  plt.show()

def chapter_graph_foundation_cricle_manifold():
  from numpy.random import default_rng
  rng = default_rng()
  vals = rng.uniform(0, 2* np.pi, 200)

  fig = plt.figure(figsize=(5,5))
  
  # Creating plot
  plt.scatter(np.cos(vals), np.sin(vals), cmap='hsv')
  plt.xticks([-1, -0.5,0,0.5,1])
  plt.yticks([-1, -0.5,0,0.5,1])

      


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

def chapter_graph_foundation_manifolds_clean():
  resolution = 200
  samples = 500
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


def chapter_graph_foundation_manifolds_noisy():
  resolution = 200
  samples = 500
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
  eVec2 = get_embedding(sino_noisy_2, 6, 2)
  _,idx2,angles2 = estimate_angles(eVec2)
  plt.scatter(eVec2[:, 0], eVec2[:, 1], s=4, c=angles2,cmap='hsv')
  plt.xticks([-0.08, -0.04, 0 , 0.04, 0.08])
  plt.yticks([-0.1, -0.05, 0 , 0.03, 0.06])
  # plt.title(f"Manifold noisy sinogram k = 4, SNR={snr}dB")

  _,idx,angles = estimate_angles(eVec_clean)
  _, fbp_clean,_ =  setup_forward_and_backward(resolution, samples, angles)
  _, fbp1,_ =  setup_forward_and_backward(resolution, samples, angles1)
  _, fbp2,_ =  setup_forward_and_backward(resolution, samples, angles2)

  reco_clean = fbp_clean(pad(torch.from_numpy(sino[idx])))
  reco_1 = fbp1(pad(torch.from_numpy(sino_noisy_1[idx1])))
  reco_2 = fbp2(pad(torch.from_numpy(sino_noisy_2[idx2])))

  plot_imshow(reco_clean, colorbar=False, size=(10,10))
  plot_imshow(reco_1, colorbar=False, size=(10,10))
  plot_imshow(reco_2, colorbar=False, size=(10,10))

  plot_imshow(fbp(pad(torch.from_numpy(sino))), colorbar=False, size=(10,10))
  plot_imshow(fbp(pad(torch.from_numpy(sino_noisy_1))), colorbar=False, size=(10,10))
  plot_imshow(fbp2(pad(torch.from_numpy(sino_noisy_2))), colorbar=False, size=(10,10))


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

import csv
def chapter_results_small_overall_compoents():
  csv_file_name = "src/small_components_knn2.csv"
  import csv

  size = 12
  results = np.zeros((size,4))
  result_names = []
  colors_list = ['lime', 'g', 'r', 'c', 'm', 'y', 'coral', 'k', 'gold', 'navy', 'crimson', 'b', 'teal']
  x = [0, -5, -10, -15]
  
  import matplotlib.colors as colors
  lots_of_colors = colors.get_named_colors_mapping()

  with open(csv_file_name, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter = ";")
    line_count = 0
    for row in csv_reader:
      if line_count == 0:
        print(f'Column names are {", ".join(row)}')
        line_count += 1

      result_names.append(row['Model/algortihm'])
      results[line_count - 1] = np.array([row['0'], row['-5'], row['-10'], row['-15']])
      line_count += 1

  fig = plt.figure(figsize=(14,7))

  for i,c in zip(range(size), colors_list):
    plt.plot(x, results[i], marker='s', color=c)

  plt.xticks([-15,-10,-5,0])
  plt.yticks([-5,0,5, 7.5, 10, 12.5])
  plt.ylim([-8, 14])
  plt.xlabel("$SNR_y$")
  plt.ylabel("$SNR$")
  plt.xlim([-16, 1])
  plt.legend(result_names)
  plt.show()

def chapter_results_large_overall_compoents():
  csv_file_name = "src/large_components_knn2.csv"
  import csv

  size = 10
  results = np.zeros((size,4))
  result_names = []

  x = [0, -5, -10, -15]
  
  with open(csv_file_name, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter = ";")
    line_count = 0
    for row in csv_reader:
      if line_count == 0:
        print(f'Column names are {", ".join(row)}')
        line_count += 1

      result_names.append(row['Model/algortihm'])
      results[line_count - 1] = np.array([row['0'], row['-5'], row['-10'], row['-15']])
      #results[line_count - 1] = np.array([row['-15'], row['-10'], row['-5'], row['0']])
      line_count += 1

  fig = plt.figure(figsize=(14,7))

  for i in range(size):
    plt.plot(x, results[i], marker='s')

  plt.xticks([-15,-10,-5,0])
  plt.yticks([-5,0,5, 7.5, 10, 12.5, 15, 17])
  plt.ylim([-8, 18])
  plt.xlabel("$SNR_y$")
  plt.ylabel("$SNR$")
  plt.xlim([-16, 1])
  plt.legend(result_names, loc='lower right')
  plt.show()

def wandb_rename(project_name : str, run_id : str, new_name : str):
  import wandb
  api = wandb.Api()

  run = api.run(f"cedric-mendelin/{project_name}/{run_id}")
  run.name = new_name
  run.update()

def wandb_export_project(project_name:str):
  import wandb
  api = wandb.Api()

  # Project is specified by <entity/project-name>
  runs = api.runs("cedric-mendelin/" + project_name)
  name_list = [] 
  snr_result = np.zeros(len(runs)) 
  loss_result = np.zeros(len(runs)) 
  duration_result = np.zeros(len(runs)) 
  config_list=[]
  counter = 0
  config_names = ['gat_snr_lower', 'gat_use_conv', 'unet_refinement', 'unet_train']
  #config_names = ['gat_layers', 'gat_heads', 'loss']
  for run in runs: 
      # run.summary are the output key/values like accuracy.
      # We call ._json_dict to omit large files 
      summary = run.summary._json_dict

      # run.config is the input metrics.
      # We remove special values that start with _.
      config = {k:v for k,v in run.config.items() if not k.startswith('_') and k in config_names}
      config_list.append(config)

      snrs = []
      loss = []
      for i, row in run.history(keys=["val_snr_reco_denoised"]).iterrows():
        snrs.append(row["val_snr_reco_denoised"])
      for i, row in run.history(keys=["val_loss_reco_denoised"]).iterrows():
        loss.append(row["val_loss_reco_denoised"])

      snr_result[counter]  = (np.mean( np.array(snrs)))
      loss_result[counter] = (np.mean( np.array(loss)))
      # duration_result[counter] = summary['training']

      # run.name is the name of the run.
      name_list.append(run.name)
      counter += 1       

  
  import pandas as pd 

  snr_df = pd.DataFrame(snr_result, columns=['SNR']) 
  loss_df = pd.DataFrame(loss_result, columns=['Loss'])
  # duration_df = pd.DataFrame(duration_result, columns=['training'])
  name_df = pd.DataFrame({'name': name_list}) 
  config_df = pd.DataFrame.from_records(config_list) 
  all_df = pd.concat([name_df, snr_df, loss_df, config_df], axis=1)

  all_df.to_csv("wandb_results/" + project_name + ".csv")
  
#chapter_imaging_sinos()
#chapter_graph_foundation_cricle_manifold()
#chapter_graph_foundation_sphere_manifold()

# chapter_graph_foundation_manifolds_different_k()
# chapter_graph_foundation_manifolds_clean()
#chapter_graph_foundation_manifolds_noisy()
# chapter_graph_foundation_manifolds_noisy()
#chapter_graph_foundation_manifolds_clean_samples_importance()
#chapter_graph_foundation_manifolds_clean_resolution_importance()
chapter_results_small_overall_compoents()
#chapter_results_large_overall_compoents()
#test_wandb_api()
#wandb_export_project("LoDoPaB-Large-knn-2")  
#wandb_rename("LoDoPaB-Large-knn-2", "1y8gw6kv", "conv_gat_unet_train_snr--15_epochs20_3")
