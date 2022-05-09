from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Dataset
from .SNRHelper import add_noise
import torch.nn.functional as F
import torch


################### GAT class ##############################
class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim, heads=1, dropout=0.5):
        super().__init__()

        #assert num_layers > 0
        # in_dim = hidden_dim * heads
        self.convs = torch.nn.ModuleList()
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

##################### Pytorch Dataset ####################
class SinogramToSinogramDataset(Dataset):
    def __init__(self, M, sinograms, noisy_sinograms, graph):
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
        self.V = V
        self.sinograms = sinograms
        self.graph = graph
        self.snr = snr

    def __len__(self):
        return self.V

    def __getitem__(self, idx):
        noisy_sinogram = add_noise(self.snr, self.sinograms[idx])
        return Data(x=noisy_sinogram, y=self.sinograms[idx], edge_index=self.graph)