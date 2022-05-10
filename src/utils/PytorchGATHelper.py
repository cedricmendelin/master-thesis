from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Dataset
from .SNRHelper import add_noise
import torch.nn.functional as F
import torch


################### GAT class ##############################
class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim, heads=1, dropout=0.5, activation=F.elu):
        """ Ctor for GAT.
        Args:
            in_dim (int): Input dimension, in the used scenario image resolution.
            hidden_dim (int): Hidden dimension. Hidden_dim x heads must be equal to in_dim.
            num_layers (int): number of layers
            out_dim (int): Number of output dimension, in the used scenario image resolution as well.
            heads (int, optional): Number of heads. Defaults to 1.
            dropout (float, optional): Dropout for every layer, except last one.. Defaults to 0.5.
            activation (func): Activation function after every layer, except last one.
        """
        super().__init__()

        # in_dim = hidden_dim * heads
        self.convs = torch.nn.ModuleList()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.dropout = dropout
        self.activation = activation

        for _ in range(num_layers - 1):
            self.convs.append(GATConv(in_dim, hidden_dim, heads))

        # last layer:
        self.convs.append(GATConv(hidden_dim * heads, out_dim, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if len(self.convs) - 1 != layer:
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