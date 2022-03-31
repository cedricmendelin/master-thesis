
from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon

from utils.Graph import *
from utils.Plotting import *
from utils.pytorch_radon.radon import *
import matplotlib.pyplot as plt

from skimage.transform import  rescale, radon, iradon
from sklearn.metrics.pairwise import haversine_distances
import torch
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from scipy.spatial import distance_matrix
import torch.nn.functional as F
import math
import imageio
import wandb

import numpy as np
from numpy.random import default_rng
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##################### Helpers ################
def add_noise(SNR, sinogram):
    VARIANCE = 10 ** (-SNR/10) * (torch.std(sinogram) **2) 
    noise = torch.randn(sinogram.shape[0], sinogram.shape[1]) * torch.sqrt(VARIANCE)
    return sinogram + noise

def estimate_angles(graph_laplacian, degree=False):
  # arctan2 range [-pi, pi]
  angles = np.arctan2(graph_laplacian[:,0],graph_laplacian[:,1]) + np.pi
  # sort idc ascending, [0, 2pi]
  idx  = np.argsort(angles)

  if degree:
    return np.degrees(angles), idx, np.degrees(angles[idx])
  else:
    return angles, idx, angles[idx]

def uniform_dist_angles(N):
    n = N //2
    from torch.distributions.uniform import Uniform
    normal = Uniform(torch.tensor([0.0]), 2 * torch.pi)
    input_angles = normal.sample((n,))
    input_angles = torch.cat([input_angles, torch.remainder(input_angles + torch.pi, 2 * torch.pi)])

    return input_angles

################## Parameters ########################
# RESOLUTION : int = 200 
# N : int = 512

SEED = 2022
torch.manual_seed(2022)
#graph
# K = 8


def run(project_name, image, image_name, snr=25, epochs=1000, layers=3, heads=2, droput=0.05, weight_decay=5e-4, lr=0.01, N=1024, K=8, RESOLUTION = 200, debug_plot = True, use_wandb = False):
    #noise
    SNR=snr
    EPOCHS = epochs
    GAT_LAYERS = layers
    GAT_HEADS = heads
    GAT_DROPOUT = droput
    GAT_ADAM_WEIGHTDECAY = weight_decay
    GAT_ADAM_LR = lr

    config = {
        "samples": N,
        "resolution": RESOLUTION,
        "noise_SNR": SNR,
        "k-nn": K,
        "gat-epochs": EPOCHS,
        "gat-layers" : GAT_LAYERS,
        "gat-heads" : GAT_HEADS,
        "gat-DROPOUT" : GAT_DROPOUT,
        "gat-adam-weightdecay" : GAT_ADAM_WEIGHTDECAY,
        "gat-adam-learningrate" : GAT_ADAM_LR,
        "name" : image_name
        }

    if use_wandb:
        wandb.init(project=project_name, entity="cedric-mendelin", config=config, reinit=True)

        wandb.run.name = f"{image_name}-{RESOLUTION}-{N}-{SNR}-{K}-{EPOCHS}-{GAT_LAYERS}-{GAT_HEADS}-{GAT_DROPOUT}-{GAT_ADAM_WEIGHTDECAY}"
        print(wandb.run.name)

    #################### Input #####################
    #input = imageio.imread('src/maps/toy_image.png')
    input = image

    scaleX = RESOLUTION/ input.shape[0]
    scaleY = RESOLUTION/ input.shape[1]
    input = rescale(input, scale=(scaleX, scaleY), mode='reflect', multichannel=False)

    input_t = torch.from_numpy(input).type(torch.float)#.to(device)
    if debug_plot:
        plot_imshow(input, title="Input image")

    if use_wandb:
        wandb.log({"input_image": wandb.Image(input)})

    ################# Angles ##########################
    from torch.distributions.uniform import Uniform


    angles_degrees =  torch.linspace(0, 360, N).type(torch.float)
    angles =  torch.linspace(0, 2 * torch.pi, N).type(torch.float)
    angles_degrees_np = np.linspace(0,360,N)
    angles_np = np.linspace(0,2 * np.pi, N)
    points_np = np.array([np.cos(angles_np), np.sin(angles_np)]).T

    ############### Forward #########################

    radon_class = Radon(RESOLUTION, angles_degrees, circle=True)

    sinogram = radon_class.forward(input_t.view(1, 1, RESOLUTION, RESOLUTION))[0,0]
    noisy_sinogram = add_noise(SNR, sinogram)

    print("Loss sinogramm noisy", torch.linalg.norm(sinogram - noisy_sinogram))
    if use_wandb:
        wandb.log({"loss_sinogramm_noisy_start": torch.linalg.norm(sinogram - noisy_sinogram)})

    x_est_GL_t = filterBackprojection2D(sinogram.T, angles_degrees)

    x_est_GL_t_noisy = filterBackprojection2D(noisy_sinogram.T, angles_degrees)

    if debug_plot:
        plot_imshow(sinogram.T, title="Sinogram uniform")
        plot_imshow(noisy_sinogram.T, title="Sinogram noisy")
        plot_imshow(x_est_GL_t, title="reconstruction")
        plot_imshow(x_est_GL_t_noisy, title="reconstruction noisy sinogram")
    
    if use_wandb:
        wandb.log({"sinogram": wandb.Image(sinogram.T)})
        wandb.log({"noisy sinogram": wandb.Image(noisy_sinogram.T)})
        wandb.log({"reconstruction": wandb.Image(x_est_GL_t)})
        wandb.log({"reconstruction noisy sinogram": wandb.Image(x_est_GL_t_noisy)})

    ############################## Distances ###########################
    # distances:
    
    distances = haversine_distances(points_np, points_np)
    graph, classes, edges = generate_knn_from_distances_with_edges(distances, 2, ordering='asc', ignoreFirst=True)
    
    # gl:
    graph_laplacian = calc_graph_laplacian(graph, embedDim=2)

    if debug_plot:
        plot_2d_scatter(graph_laplacian, title=f"GL reconstruction angles K=2")

    if use_wandb:
        wandb.log({"clean GL" : wandb.plot.scatter(wandb.Table(data=graph_laplacian, columns = ["x", "y"]),"x","y", title=f"GL sinogram")})

    # gl_angles, gl_idx, gl_angles_sotrted = estimate_angles(graph_laplacian)
    # gl_idx = np.argsort(gl_angles)
    # gl_idx_t = torch.tensor(gl_idx).type(torch.long)
    # x_est_GL_t = filterBackprojection2D(sinogram.T[gl_idx], reconstruction_angles_degrees)
    # plot_imshow(x_est_GL_t, title="reconstruction from GL angles sorting")

    ################### GAT class #############################
    class GAT(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, num_layers, out_dim, heads = 1, dropout=0.5):
            super().__init__()
            
            #assert num_layers > 0
            # in_dim = hidden_dim * heads
            self.convs =  torch.nn.ModuleList()
            self.in_dim = in_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.out_dim = out_dim
            self.dropout = dropout

            # layer 1:
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(in_dim, hidden_dim, heads))

            # last layer:
            self.convs.append(GATConv(hidden_dim * heads, out_dim, 1))

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            for layer, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if len(self.convs) - 1 != layer:
                    x = F.elu(x)
                    x = F.dropout(x, self.dropout)

            return x

    ################ GAT #################
    # prep edges:
    N_noisy_edges = len(edges)
    edge_index = np.zeros((2, N_noisy_edges))
    noisy_edges_list = list(edges)
    for i in range(N_noisy_edges):
        (n,m) = noisy_edges_list[i]
        edge_index[0,i] = n
        edge_index[1,i] = m

    # GAT setup:
    # def __init__(self, in_dim, hidden_dim, num_layers, out_dim, heads = 1, dropout=0.5):
    model_sinogram = GAT(
        in_dim=RESOLUTION, 
        hidden_dim=RESOLUTION // GAT_HEADS,
        num_layers=GAT_LAYERS, 
        out_dim=RESOLUTION,  
        heads=GAT_HEADS, 
        dropout=GAT_DROPOUT).to(device)

    optimizer_sinogram = torch.optim.Adam(model_sinogram.parameters(), lr=GAT_ADAM_LR, weight_decay=GAT_ADAM_WEIGHTDECAY)

    model_sinogram.train()
    
    t_edge_index = torch.tensor(edge_index.copy()).type(torch.long).to(device)

    for epoch in range(EPOCHS):
        radon_class = Radon(RESOLUTION, torch.rad2deg(uniform_dist_angles(N)).type(torch.float), circle=True)
        sinogram = radon_class.forward(input_t.view(1, 1, RESOLUTION, RESOLUTION))[0,0]
        t_y = sinogram.T.to(device)
        snr = torch.randint(5, 30, (1,))
        t_x = add_noise(snr, sinogram.clone()).T.to(device)
        
        data_sinogram = Data(x=t_x, y=t_y, edge_index=t_edge_index)

        #model_sinogram.train()

        optimizer_sinogram.zero_grad()
        out_sinogram = model_sinogram(data_sinogram)
        loss_sinogram = torch.linalg.norm(out_sinogram - data_sinogram.y)

        print(f"epoch: {epoch} --- loss sino: {loss_sinogram} with snr: {snr}")
        loss_sinogram.backward()
        
        optimizer_sinogram.step()

    model_sinogram.eval()

    for snr in range(-5,30, 3):
        uniform_angles = uniform_dist_angles(N)
        uniform_angles_degrees = torch.rad2deg(uniform_angles).type(torch.float)
        radon_class = Radon(RESOLUTION, uniform_angles_degrees , circle=True)
        sinogram = radon_class.forward(input_t.view(1, 1, RESOLUTION, RESOLUTION))[0,0]
        t_y = sinogram.T.to(device)
        t_x = add_noise(snr, sinogram.clone()).T.to(device)

        sorted, idx = torch.sort(uniform_angles_degrees, 0)
        idx = idx.view(N)

        pred_sinogram = model_sinogram(Data(x=t_x, y=t_y, edge_index=t_edge_index))
        loss_sinogram = torch.linalg.norm(pred_sinogram - t_y)
        print(f"Evaluation loss: snr {snr}, loss - {loss_sinogram}")

        denoised_reconstruction = filterBackprojection2D(pred_sinogram[idx], angles_degrees)
        noisy_reconstruction = filterBackprojection2D(t_x[idx], angles_degrees)

        if debug_plot:
            plot_imshow(denoised_reconstruction.cpu().detach().numpy(), title=f"reconstruction denoised - SNR: {snr} ")
            plot_imshow(noisy_reconstruction.cpu().detach().numpy(), title=f"reconstruction noisy - SNR: {snr} ")

    # plot_imshow(pred_sinogram.cpu().detach().numpy(), title='Denoised sinogram unsorted', c_map=color_map)



    

layer_list = [2,3,4,5,6]
head_list = [1,10,20]
epoch_list = [800,1000,1200,1400,1600,1800,2000]
#snr_list = [25,20,15,10]
dropout_list = [0.0, 0.03, 0.05, 0.08, 0.1, 0.2, 0.5, 0.6]
weight_decay_list = [5e-5, 5e-4, 5e-3,5e-2,5e-1, 1e-5, 1e-4, 1e-3,1e-2, 1e-1,1]
snr_list = [2,25]
k_list = [4,5,8,11,12]

#run(snr=25, epochs=1000, layers=3, heads=2, droput=0.05, weight_decay=5e-4, lr=0.01, N=512, K=8, RESOLUTION = 200):
# from utils.CoCoDataset import *
# image_path = "src/data/val2017/"
# image_count = 5
# files = os.listdir(image_path)
# x = load_images_files(image_path, files, 200,200 , number=image_count, num_seed=5)
# names = np.array([f"image-{i}" for i in range(image_count) ])
# for i in range(image_count):
#     head_list = [1,10,20]
#     for heads in head_list:
#         run("denoise-sinogram-gat-coco-images",x[i], names[i], epochs=1600, layers=4, heads=heads, snr=15)

# head_list = [1,20]
# # plot_imshow(x[0].reshape((RESOLUTION, RESOLUTION)))
# #x = shepp_logan_phantom()
# # names = np.array([f"image-{i}" for i in range(n_imgs) ])
# for layers in layer_list:
#     for heads in head_list:
#         for snr in snr_list:
#             run("denoise-sinogram-gat-snr",shepp_logan_phantom(), "phantom", epochs=1600, layers=layers, heads=heads, snr=snr)

run("denoise-sinogram-gat-architecutre", shepp_logan_phantom(), "phantom",N=1024, epochs=1600, layers=3, heads=10, snr=25, K=9, use_wandb=False, debug_plot=True)

# for layers in layer_list:
#     for heads in head_list:
#         for snr in snr_list:
            
            

# x_est_GL_t = filterBackprojection2D(pred_sinogram, reconstruction_angles_degrees)
# plot_imshow(x_est_GL_t.cpu().detach().numpy(), title="reconstruction denoised unsorted")

# # estimate new angles:
# # distances:
# pred_distances = distance_matrix(pred_sinogram.cpu().detach(), pred_sinogram.cpu().detach())
# pred_distances /= pred_distances.max()
# # k-nn:
# graph, classes, edges = generate_knn_from_distances_with_edges(pred_distances, K, ordering='asc', ignoreFirst=True)
# graph_laplacian_denoise = calc_graph_laplacian(graph, embedDim=2)

# plot_2d_scatter(graph_laplacian_denoise, title="Denoised Sinogram GL")
# for k in range(5, 12):
#     K = k


# gl_angles, gl_idx, gl_angles_sotrted = estimate_angles(graph_laplacian_denoise)

# gl_idx = np.argsort(gl_angles)
# # gl_idx_t = torch.tensor(gl_idx).type(torch.long)
# print(pred_sinogram.shape)
# print(gl_idx.shape)

# x_est_GL_t = filterBackprojection2D(pred_sinogram.cpu().detach()[gl_idx], reconstruction_angles_degrees)
# plot_imshow(x_est_GL_t, title="reconstruction from GL angles sorting")




plt.show()

