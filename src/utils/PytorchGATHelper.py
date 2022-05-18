from torch_geometric.nn import GATConv
from torch.nn import Conv1d
from torch_geometric.data import Data, Dataset
from .SNRHelper import add_noise
import torch.nn.functional as F
import torch


################### GAT class ##############################
class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim,num_layers, heads=1, dropout=0.5, activation=F.elu, add_conv_before_gat=True, conv_kernel=3, conv_padding=1, conv_N_latent=1):
        """ Ctor for GAT.
        Args:
            in_dim (int): Input dimension, in the used scenario image resolution.
            num_layers (int): number of layers
            out_dim (int): Number of output dimension, in the used scenario image resolution as well.
            heads (int, optional): Number of heads. Defaults to 1.
            dropout (float, optional): Dropout for every layer, except last one.. Defaults to 0.5.
            activation (func): Activation function after every layer, except last one.
        """
        super().__init__()

        self.gat_layers = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.in_dim = in_dim
        self.hidden_dim = in_dim // heads
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.dropout = dropout
        self.activation = activation
        self.use_conv = add_conv_before_gat
        self.conv_N_latent = conv_N_latent

        # setup GAT:
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(in_dim, self.hidden_dim, heads))
        
        # last layer GAT:
        # GAT averaging with one head:
        self.gat_layers.append(GATConv(self.hidden_dim * heads, out_dim, 1))

        # setup convs:
        if self.use_conv:
            # first layer:
            self.convs.append(Conv1d(in_channels=1, out_channels=self.conv_N_latent, kernel_size=conv_kernel, padding=conv_padding))
            
            # layer 2 - (n-1)
            if num_layers > 2:
                for _ in range(1, num_layers - 1):
                    self.convs.append(Conv1d(in_channels=self.conv_N_latent, out_channels=self.conv_N_latent, kernel_size=conv_kernel, padding=conv_padding))
            # last layer: 
            self.convs.append(Conv1d(in_channels=self.conv_N_latent, out_channels=1, kernel_size=conv_kernel, padding=conv_padding))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i in range(self.num_layers):
            layer = self.gat_layers[i]

            if self.use_conv:
                conv = self.convs[i]
                x = conv(x.view(x.size(dim=0), self.conv_N_latent, x.size(dim=1)))
                x = x.view(x.size(dim=0) * x.size(dim=1), x.size(dim=2))

            x = layer(x, edge_index)
            
            # do not do activation and dropout in last GAT-layer
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, self.dropout)

        return x

##################### Pytorch Dataset ####################
class SinogramToSinogramDataset(Dataset):
    def __init__(self, M, sinograms, noisy_sinograms, graph):
        """ Ctor for pytorch Dataset where x=noisy_sinogram, y=clean_sinogram, edge_index=graph.
        Args:
            M (int): Number of elements in the dataset.
            sinograms: Input sinograms.
            noisy_sinograms: Noisy sinograms.
            graph: Input graph as edge list.
        """
        self.M = M
        self.sinograms = sinograms
        self.noisy_sinograms = noisy_sinograms
        self.graph = graph
    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        return Data(x=self.noisy_sinograms[idx], y=self.sinograms[idx], edge_index=self.graph)

class SinogramToImageDataset(Dataset):
    def __init__(self, M, clean_image, noisy_sinograms, graph):
        """ Ctor for pytorch Dataset where x=noisy_sinogram, y=clean_image, edge_index=graph.
        Args:
            M (int): Number of elements in the dataset.
            clean_image: Clean original images.
            noisy_sinograms: Noisy sinograms.
            graph: Input graph as edge list.
        """
        self.M = M
        self.clean_image = clean_image
        self.noisy_sinograms = noisy_sinograms
        self.graph = graph

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        return Data(x=self.noisy_sinograms[idx], y=self.clean_image[idx], edge_index=self.graph)

class ValidationDataset(Dataset):
    def __init__(self, V, snr, sinograms, graph):
        """ Ctor for pytorch Dataset where x=noisy_sinogram, y=clean_sinogram, edge_index=graph.
        Args:
            V (int): Number of elements in the dataset.
            snr (int): SNR of noisy_sinograms.
            sinograms: Input sinograms.
            graph: Input graph as edge list.
        """
        self.V = V
        self.sinograms = sinograms
        self.graph = graph
        self.snr = snr

    def __len__(self):
        return self.V

    def __getitem__(self, idx):
        noisy_sinogram = add_noise(self.snr, self.sinograms[idx])
        return Data(x=noisy_sinogram, y=self.sinograms[idx], edge_index=self.graph)