
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
from torch_geometric.data.batch import Batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##################### Helpers ################
def add_noise(SNR, sinogram):
    VARIANCE = 10 ** (-SNR/10) * (torch.std(sinogram) **2) 
    noise = torch.randn(sinogram.shape[0], sinogram.shape[1]) * torch.sqrt(VARIANCE)
    return sinogram + noise

def add_noise_to_sinograms(sinograms, snr):
    noisy_sinograms = torch.empty_like(sinograms)
    for i in range(sinograms.shape[0]):
        noisy_sinograms[i,0] = add_noise(snr, sinograms[i,0])

    return noisy_sinograms
    

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

def find_SNR(ref, x):
    dif = torch.sum((ref-x)**2)
    nref = torch.sum(ref**2)
    return 10*torch.log10((nref+1e-16)/(dif+1e-16))

################## Parameters ########################
# RESOLUTION : int = 200 
# N : int = 512

SEED = 2022
torch.manual_seed(2022)
#graph
# K = 8


def run(project_name, images, validation_images, validation_snr=[-5,2,10,25], snr_lower=0, snr_upper=25,K =2, epochs=1000, layers=3, heads=2, droput=0.05, weight_decay=5e-4, lr=0.01, N=1024, RESOLUTION = 200, debug_plot = True, use_wandb = False):
    #noise
    # SNR=snr
    EPOCHS = epochs
    GAT_LAYERS = layers
    GAT_HEADS = heads
    GAT_DROPOUT = droput
    GAT_ADAM_WEIGHTDECAY = weight_decay
    GAT_ADAM_LR = lr
    M = images.shape[0]
    V = validation_images.shape[0]

    config = {
        "samples": N,
        "resolution": RESOLUTION,
        # "noise_SNR": SNR,
        # "k-nn": K,
        "gat-epochs": EPOCHS,
        "gat-layers" : GAT_LAYERS,
        "gat-heads" : GAT_HEADS,
        "gat-DROPOUT" : GAT_DROPOUT,
        "gat-adam-weightdecay" : GAT_ADAM_WEIGHTDECAY,
        "gat-adam-learningrate" : GAT_ADAM_LR,
        "gat_snr_lower_bound" : snr_lower,
        "gat_snr_upper_bound" : snr_upper,
        "validation_snr" : validation_snr,
        # "name" : image_name
        }

    if use_wandb:
        wandb.init(project=project_name, entity="cedric-mendelin", config=config, reinit=True)

        wandb.run.name = f"toy_images-{RESOLUTION}-{N}-{M}-{K}-{EPOCHS}-{GAT_LAYERS}-{GAT_HEADS}-{GAT_DROPOUT}-{GAT_ADAM_WEIGHTDECAY}"
        print(wandb.run.name)

    #################### Input #####################
    #input = imageio.imread('src/maps/toy_image.png')
    input = images

    input_t = torch.from_numpy(input).type(torch.float)#.to(device)
    validation_t = torch.from_numpy(input).type(torch.float)
    if debug_plot:
        for i in range(M):
            plot_imshow(input[i], title=f"Input image - {i}")

    if use_wandb:
         wandb.log({"input images": [wandb.Image(img) for img in images]})
         wandb.log({"validation images": [wandb.Image(val_img) for val_img in validation_images]})

    ################# Angles ##########################
    # from torch.distributions.uniform import Uniform


    angles_degrees =  torch.linspace(0, 360, N).type(torch.float)
    angles =  torch.linspace(0, 2 * torch.pi, N).type(torch.float)
    angles_degrees_np = np.linspace(0,360,N)
    angles_np = np.linspace(0,2 * np.pi, N)
    points_np = np.array([np.cos(angles_np), np.sin(angles_np)]).T

    ################### Forward ##############################
    radon_class = Radon(RESOLUTION, angles_degrees, circle=True)
    sinograms = radon_class.forward(input_t.view(M, 1, RESOLUTION, RESOLUTION))

    # todo: print clean reconstructionsimages

    ############################## Distances ###########################
    # distances:
    
    distances = haversine_distances(points_np, points_np)
    graph, classes, edges = generate_knn_from_distances_with_edges(distances, K, ordering='asc', ignoreFirst=True)
    graph_laplacian = calc_graph_laplacian(graph, embedDim=2)

    if debug_plot:
        plot_2d_scatter(graph_laplacian, title=f"GL clean case K={K}")

    if use_wandb:
        wandb.log({"clean GL" : wandb.plot.scatter(wandb.Table(data=graph_laplacian, columns = ["x", "y"]),"x","y", title=f"GL sinogram")})


    N_noisy_edges = len(edges)
    edge_index = np.zeros((2, N_noisy_edges))
    noisy_edges_list = list(edges)
    for i in range(N_noisy_edges):
        (n,m) = noisy_edges_list[i]
        edge_index[0,i] = n
        edge_index[1,i] = m
   
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
        snr = torch.randint(snr_lower, snr_upper, (1,))
        noisy_sinograms = add_noise_to_sinograms(sinograms, snr)
        # radon_class = Radon(RESOLUTION, torch.rad2deg(uniform_dist_angles(N)).type(torch.float), circle=True)
        # sinogram = radon_class.forward(input_t.view(1, 1, RESOLUTION, RESOLUTION))[0,0]
        # t_x = noisy_sinograms.to(device)

        # print(sinograms.shape)
        # print(noisy_sinograms.shape)
        loss = 0
        data = []
        for i in range(M):
            data = Data(x=noisy_sinograms[i,0].T.to(device), y=sinograms[i,0].T.to(device), edge_index=t_edge_index)
            optimizer_sinogram.zero_grad()

            out_sinograms = model_sinogram(data)
            loss_sinogram = torch.linalg.norm(out_sinograms - data.y)

            print(f"epoch: {epoch} img: {i} --- loss sino: {loss_sinogram} with snr: {snr}")
            
            loss += loss_sinogram
            loss_sinogram.backward()
            optimizer_sinogram.step()
        
        if use_wandb:
            wandb.log({
                "epoch" : epoch,
                "loss" : loss,
            })

    model_sinogram.eval()

    # validation_snr = [-5,2,10,25]
    validation_t = torch.from_numpy(validation_images).type(torch.float)#.to(device)
    radon_class = Radon(RESOLUTION, angles_degrees , circle=True)
    validation_sinograms = radon_class.forward(validation_t.view(V, 1, RESOLUTION, RESOLUTION))

    validation_index = 0

    for snr in validation_snr:
        for i in range(V):
            t_y = validation_sinograms[i,0].T.to(device)
            t_x = add_noise(snr, validation_sinograms[i,0].clone()).T.to(device)
            pred_sinogram = model_sinogram(Data(x=t_x, y=t_y, edge_index=t_edge_index))
            loss_sinogram = torch.linalg.norm(pred_sinogram - t_y)
            loss_sinogram_noisy = torch.linalg.norm(t_x - t_y)

            denoised_reconstruction = filterBackprojection2D(pred_sinogram, angles_degrees)
            noisy_reconstruction = filterBackprojection2D(t_x, angles_degrees)

            if use_wandb:
                wandb.log({
                    "val_index" : validation_index,
                    "val_input_snr":snr,
                    "val_loss" : loss_sinogram,
                    "val_loss_noisy": loss_sinogram_noisy,
                    "val_input_snr_calculated": find_SNR(t_y, t_x),
                    "val_denoised_snr_calculated": find_SNR(t_y, pred_sinogram),
                    "val_denoised_reconstruction" : wandb.Image(denoised_reconstruction.cpu().detach().numpy(), caption=f"Rec denoised - SNR: {snr} - image {i}"),
                    "val_noisy reconstruction" : wandb.Image(noisy_reconstruction.cpu().detach().numpy(), caption=f"Rec noisy - SNR: {snr} - image {i}"),
                })
                validation_index += 1
            if debug_plot:
                reconstructions = np.array([denoised_reconstruction.cpu().detach().numpy(), noisy_reconstruction.cpu().detach().numpy()])
                titles = [f"Rec denoised - SNR: {snr} - image {i}", f"Rec noisy - SNR: {snr} - image {i}" ]
                plot_image_grid(reconstructions, titles)
    
    if use_wandb:
        wandb.finish()

    return model_sinogram

    



from utils.CoCoDataset import *
import time
t = time.time()
image_path = "src/toyimages/64/"
files = os.listdir(image_path)

N = 1024
RESOLUTION = 64

count_list = [1,2,3,4,5,6,7,8]

x = load_images_files(image_path, files, 64, 64 , number=10, num_seed=5)

for image_count in count_list:
    validation_image_count = 2
    x_input = x[0:image_count]
    x_validation = x[image_count: image_count+validation_image_count]
    #def run(project_name, images, validation_images, validation_snr, snr_lower=0, snr_upper=25,K =2, epochs=1000, layers=3, heads=2, droput=0.05, weight_decay=5e-4, lr=0.01, N=1024, RESOLUTION = 200, debug_plot = True, use_wandb = False):
    _ = run("toy-images-input-size", x_input, x_validation, validation_snr=[-5,2,10,25], snr_lower=-5, snr_upper=30, epochs=2000, layers=3, heads=4, droput=0.05, weight_decay=5e-4, lr=0.01, N=1024, RESOLUTION = RESOLUTION, debug_plot = False, use_wandb = True)

heads_list = [1,2,4,8,16]
count_list = [4,5,6]
for image_count in count_list:
    for gat_heads in heads_list:
        validation_image_count = 2
        x_input = x[0:image_count]
        x_validation = x[image_count: image_count+validation_image_count]
        #def run(project_name, images, validation_images, validation_snr, snr_lower=0, snr_upper=25,K =2, epochs=1000, layers=3, heads=2, droput=0.05, weight_decay=5e-4, lr=0.01, N=1024, RESOLUTION = 200, debug_plot = True, use_wandb = False):
        _ = run("toy-images-heads", x_input, x_validation, validation_snr=[-5,2,10,25], snr_lower=-5, snr_upper=30, epochs=2000, layers=3, heads=gat_heads, droput=0.05, weight_decay=5e-4, lr=0.01, N=1024, RESOLUTION = RESOLUTION, debug_plot = False, use_wandb = True)

        
print("execution time", time.time() - t)

plt.show()

