
from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon

from utils.Graph import *
from utils.Plotting import *
from utils.pytorch_radon.radon import *
import matplotlib.pyplot as plt

from skimage.transform import  rescale, radon, iradon

import torch
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from scipy.spatial import distance_matrix
import torch.nn.functional as F
import math
import imageio
import wandb

import numpy as np
from numpy.random import default_rng

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
################## Parameters ########################
# RESOLUTION : int = 200 
# N : int = 512

SEED = 2022
torch.manual_seed(2022)
#graph
# K = 8


def run(project_name, image, image_name, snr=25, epochs=1000, layers=3, heads=2, droput=0.05, weight_decay=5e-4, lr=0.01, N=512, K=8, RESOLUTION = 200, debug_plot = True, use_wandb = False):
    #noise
    SNR=snr
    EPOCHS = epochs
    GAT_LAYERS = layers
    GAT_HEADS = heads
    GAT_DROPOUT = droput
    GAT_ADAM_WEIGHTDECAY = weight_decay
    GAT_ADAM_LR = lr

    config = {
        "samples": N * 2,
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
    normal = Uniform(torch.tensor([0.0]), 2 * torch.pi)
    input_angles = normal.sample((N,))
    input_angles = torch.cat([input_angles, torch.remainder(input_angles + torch.pi, 2 * torch.pi)])
    N = N * 2

    input_angles_degrees = torch.rad2deg(input_angles).type(torch.float)
    x = torch.cos(input_angles)
    y = torch.sin(input_angles)
    p = torch.stack((x,y)).T
    print(p.shape)
    plot_2d_scatter( p.view(1024, 2).numpy())


    reconstruction_angles_degrees =  torch.linspace(0, 360, N).type(torch.float)
    reconstruction_angles =  torch.linspace(0, 2 * torch.pi, N).type(torch.float)

    ############### Forward #########################

    radon_class = Radon(RESOLUTION, input_angles_degrees, circle=True)

    sinogram = radon_class.forward(input_t.view(1, 1, RESOLUTION, RESOLUTION))[0,0]
    noisy_sinogram = add_noise(SNR, sinogram)

    print("Loss sinogramm noisy", torch.linalg.norm(sinogram - noisy_sinogram))
    if use_wandb:
        wandb.log({"loss_sinogramm_noisy_start": torch.linalg.norm(sinogram - noisy_sinogram)})


    sorted, idx = torch.sort(input_angles_degrees, 0)
    idx = idx.view(N)

    x_est_GL_t = filterBackprojection2D(sinogram.T[idx], reconstruction_angles_degrees)

    x_est_GL_t_noisy = filterBackprojection2D(noisy_sinogram.T[idx], reconstruction_angles_degrees)

    if debug_plot:
        plot_imshow(sinogram.T[idx], title="Sinogram uniform")
        plot_imshow(noisy_sinogram.T[idx], title="Sinogram noisy")
        plot_imshow(x_est_GL_t, title="reconstruction")
        plot_imshow(x_est_GL_t_noisy, title="reconstruction noisy sinogram")
    
    if use_wandb:
        wandb.log({"sinogram": wandb.Image(sinogram.T[idx])})
        wandb.log({"noisy sinogram": wandb.Image(noisy_sinogram.T[idx])})
        wandb.log({"reconstruction": wandb.Image(x_est_GL_t)})
        wandb.log({"reconstruction noisy sinogram": wandb.Image(x_est_GL_t_noisy)})

    ############################## Distances ###########################
    # distances:
    distances = distance_matrix(sinogram.T, sinogram.T)
    distances /= distances.max()

    noisy_distances = distance_matrix(noisy_sinogram.T, noisy_sinogram.T)
    noisy_distances /= noisy_distances.max()

    # k-nn:

    # for k in range(5, 12):
    #     K = k
    graph, classes, edges = generate_knn_from_distances_with_edges(distances, K, ordering='asc', ignoreFirst=True)
    noisy_graph, noisy_classes, noisy_edges = generate_knn_from_distances_with_edges(noisy_distances, K, ordering='asc', ignoreFirst=True)

    # gl:
    graph_laplacian = calc_graph_laplacian(graph, embedDim=2)
    noisy_graph_laplacian = calc_graph_laplacian(noisy_graph, embedDim=2)

    _, noisy_gl_idx, _ = estimate_angles(noisy_graph_laplacian)
    t_gl_idx = torch.from_numpy(noisy_gl_idx).type(torch.long)

    x_est_GL_t_noisy = filterBackprojection2D(noisy_sinogram.T[t_gl_idx], reconstruction_angles_degrees)
    plot_imshow(x_est_GL_t_noisy, title="reconstruction noisy sinogram GL angles")

    if debug_plot:
        plot_2d_scatter(graph_laplacian, title=f"GL clean case K={K}")
        plot_2d_scatter(noisy_graph_laplacian, title="GL noisy case")

    if use_wandb:
        wandb.log({"noisy GL" : wandb.plot.scatter(wandb.Table(data=noisy_graph_laplacian, columns = ["x", "y"]),"x","y", title=f"GL noisy sinogram")})
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

    class MLP(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim,  out_dim, dropout=0.5):
            super().__init__()
            
            #assert num_layers > 0
            # in_dim = hidden_dim * heads
            self.convs =  torch.nn.ModuleList()
            self.in_dim = in_dim
            self.hidden_dim = hidden_dim
            self.out_dim = out_dim
            self.dropout = dropout

            # layer 1:
            self.convs.append(torch.nn.Linear(self.in_dim, self.hidden_dim))
            # self.convs.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))

            # last layer:
            self.convs.append(torch.nn.Linear(hidden_dim , out_dim))

        def forward(self, data):
            x =  data.x
            for layer, conv in enumerate(self.convs):
                x = conv(x)
                if len(self.convs) - 1 != layer:
                    x = F.elu(x)
                    x = F.dropout(x, self.dropout)

            return x

    ################ GAT #################
    # prep edges:
    N_noisy_edges = len(noisy_edges)
    edge_index = np.zeros((2, N_noisy_edges))
    noisy_edges_list = list(noisy_edges)
    for i in range(N_noisy_edges):
        (n,m) = noisy_edges_list[i]
        edge_index[0,i] = n
        edge_index[1,i] = m

    # GAT setup:
    t_y = sinogram.T.clone().to(device)
    print("t_y shape", t_y.shape)
    t_x = noisy_sinogram.T.clone().to(device)
    t_edge_index = torch.tensor(edge_index.copy()).type(torch.long).to(device)

    from torch_geometric.data import Data


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data_sinogram = Data(x=t_x, y=t_y, edge_index=t_edge_index, edge_attr=t_edge_attribute)
    data_sinogram = Data(x=t_x, y=t_y, edge_index=t_edge_index)

    # def __init__(self, in_dim, hidden_dim, num_layers, out_dim, heads = 1, dropout=0.5):
    model_sinogram = MLP(
        in_dim=RESOLUTION, 
        hidden_dim=RESOLUTION // GAT_HEADS,
        out_dim=RESOLUTION,  
        dropout=GAT_DROPOUT).to(device)

    optimizer_sinogram = torch.optim.Adam(model_sinogram.parameters(), lr=GAT_ADAM_LR, weight_decay=GAT_ADAM_WEIGHTDECAY)

    model_sinogram.train()

    for epoch in range(EPOCHS):
        #model_sinogram.train()

        optimizer_sinogram.zero_grad()
        out_sinogram = model_sinogram(data_sinogram)
        loss_sinogram = torch.linalg.norm(out_sinogram - data_sinogram.y)
        if epoch % 50 == 0 and use_wandb:
            loss_sinogram_np = loss_sinogram.clone().cpu().detach()
            epoch_out_np = out_sinogram[t_gl_idx].clone().cpu().detach().numpy()

            wandb.log({
                "epoch" : epoch,
                "loss" : loss_sinogram_np,
                "out_sinogram" :  wandb.Image(epoch_out_np, caption=epoch),
                "out_reconstructed" : wandb.Image(filterBackprojection2D(out_sinogram[t_gl_idx], reconstruction_angles_degrees).cpu().detach().numpy(), caption=epoch)
            })
            

        print(f"epoch: {epoch} --- loss sino: {loss_sinogram} ")
        loss_sinogram.backward()
        
        optimizer_sinogram.step()

    model_sinogram.eval()
    pred_sinogram = model_sinogram(data_sinogram)

    # plot_imshow(pred_sinogram.cpu().detach().numpy(), title='Denoised sinogram unsorted', c_map=color_map)

    x_est_GL_t = filterBackprojection2D(pred_sinogram[t_gl_idx], reconstruction_angles_degrees)

    if debug_plot:
        plot_imshow(pred_sinogram[t_gl_idx].cpu().detach().numpy(), title='Denoised sinogram sorted', c_map=color_map)
        plot_imshow(x_est_GL_t.cpu().detach().numpy(), title="reconstruction denoised sorted ")
    
    if use_wandb:
        wandb.log({"Denoised sinogram": wandb.Image(pred_sinogram[idx].cpu().detach().numpy())})
        wandb.log({"Reconstruction denoised sinogram": wandb.Image(x_est_GL_t.cpu().detach().numpy())})

        wandb.log({"loss_sinogramm_noisy_end": torch.linalg.norm(pred_sinogram.cpu() - sinogram.T)})

        wandb.finish()

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

run("denoise-sinogram-gat-architecutre", shepp_logan_phantom(), "phantom", epochs=1600, layers=3, heads=1, snr=5,K=8, use_wandb=False, debug_plot=True)

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

