from sklearn.metrics.pairwise import haversine_distances

from .ODLHelper import OperatorFunction, OperatorModule


from .Graph import *
from .pytorch_radon.radon import *
from .Plotting import *
import numpy as np
import wandb
import time
from torch_geometric.nn import GATConv, DataParallel
from torch_geometric.loader import DataListLoader
from torch_geometric.data import Data, Dataset
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2

import odl

######################### static helpers ######################


def _add_noise(SNR, sinogram):
    VARIANCE = 10 ** (-SNR/10) * (torch.std(sinogram) ** 2)
    noise = torch.randn(
        sinogram.shape[0], sinogram.shape[1]) * torch.sqrt(VARIANCE)
    return sinogram + noise


def _add_noise_to_sinograms(sinograms, snr):
    noisy_sinograms = torch.empty_like(sinograms)
    for i in range(sinograms.shape[0]):
        noisy_sinograms[i] = _add_noise(snr, sinograms[i])

    return noisy_sinograms


def _find_SNR(ref, x):
    dif = torch.sum((ref-x)**2)
    nref = torch.sum(ref**2)
    return 10*torch.log10((nref+1e-16)/(dif+1e-16))


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

#################### Sinogram custom Dataset ###############
class CustomSinogramDataset(Dataset):
    def __init__(self, M, clean_image, noisy_sinograms, graph):
        self.M = M
        self.clean_image = clean_image
        self.noisy_sinograms = noisy_sinograms
        self.graph = graph

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        return Data(x=self.noisy_sinograms[idx], y=self.clean_image[idx], edge_index=self.graph)

class CustomValidationSinogramDataset(Dataset):
    def __init__(self, V, snr, sinograms, clean_image, graph):
        self.V = V
        self.sinograms = sinograms
        self.clean_image = clean_image
        self.graph = graph
        self.snr = snr

    def __len__(self):
        return self.V

    def __getitem__(self, idx):
        noisy_sinogram = _add_noise(self.snr, self.sinograms[idx])
        return Data(x=noisy_sinogram, y=self.clean_image[idx], edge_index=self.graph)

################### Denoiser class #########################
class GatDenoiserEndToEnd():
    def __init__(
        self,
        args,
        type
    ):
        self.args = args
        self.VERBOSE = args.verbose
        self.loader = None
        self.type = type

        self.USE_WANDB = args.use_wandb
        self.DEBUG_PLOT = args.debug_plots

        if self.VERBOSE:
            self.time_dict = {}

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.device0 = self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def create_validator(cls, args, model_state):
        validator =  GatDenoiserEndToEnd(args, 'validator')
        validator.__initialize_validator__(args, model_state)
        return validator

    def __initialize_validator__(self, args, model_state):
        self.N: int = args.samples
        self.RESOLUTION: int = args.resolution
        self.GAT_LAYERS: int = args.gat_layers
        self.GAT_HEADS: int = args.gat_heads
        self.GAT_DROPOUT: float = args.gat_dropout
        self.K : int = args.k_nn

        self.__prepare_angles__()
        self.__prepare_graph__()

        self.model_sinogram = GAT(
            in_dim=self.RESOLUTION,
            hidden_dim=self.RESOLUTION // self.GAT_HEADS,
            num_layers=self.GAT_LAYERS,
            out_dim=self.RESOLUTION,
            heads=self.GAT_HEADS,
            dropout=self.GAT_DROPOUT)

        self.model_sinogram.load_state_dict(model_state)

        self.model_sinogram = DataParallel(self.model_sinogram)
        self.model_sinogram.to(self.device)

    @classmethod
    def create(cls, args, input_images):
        denoiser =  GatDenoiserEndToEnd(args, 'denoiser')
        denoiser.__initialize_denoiser__(input_images, args)
        return denoiser

    def __initialize_denoiser__(self, input_images, args):
        self.INPUT_IMAGES: list = input_images
        self.M: int = input_images.shape[0]
        self.N: int = args.samples
        self.RESOLUTION: int = args.resolution
        self.K: int = args.k_nn
        self.EPOCHS: int = args.epochs
        self.GAT_LAYERS: int = args.gat_layers
        self.GAT_HEADS: int = args.gat_heads
        self.GAT_DROPOUT: float = args.gat_dropout
        self.GAT_ADAM_WEIGHTDECAY: float = args.gat_weight_decay
        self.GAT_ADAM_LR: float = args.gat_learning_rate

        if args.gat_snr_upper is None or args.gat_snr_lower == args.gat_snr_upper:
            if self.VERBOSE:
                print("Using fixed snr!")
            self.SNR = args.gat_snr_lower
            self.FIXED_SNR = True
            self.SNR_LOWER = args.gat_snr_lower
            self.SNR_UPPER = args.gat_snr_upper
        else:
            self.FIXED_SNR = False
            self.SNR_LOWER = args.gat_snr_lower
            self.SNR_UPPER = args.gat_snr_upper

        if self.USE_WANDB:
            self.init_wandb(args.wandb_project, args)

        self.__execute_and_log_time__(
            lambda: self.__prepare_images__(), "prep_images")
        self.__execute_and_log_time__(
            lambda: self.__prepare_angles__(), "prep_angles")
        self.__execute_and_log_time__(lambda: self.__forward__(), "forward")
        self.__execute_and_log_time__(lambda: self.__prepare_graph__(), "prep_graph")
        self.__execute_and_log_time__(lambda: self.__prepare_model__(), "prep_model")

    

    def __prepare_images__(self):
        self.T_input_images = torch.from_numpy(
            self.INPUT_IMAGES).type(torch.float)
        if self.DEBUG_PLOT:
            if self.M < 10:
                for i in range(self.M):
                    plot_imshow(
                        self.INPUT_IMAGES[i], title=f"Input image - {i}")
            else:
                for i in range(10):
                    plot_imshow(
                        self.INPUT_IMAGES[i], title=f"Input image - {i}")

        if self.USE_WANDB:
            if self.M < 10:
                wandb.log({"input images": [wandb.Image(img)
                                            for img in self.INPUT_IMAGES]})
            else:
                wandb.log({"input images": [wandb.Image(img)
                                            for img in self.INPUT_IMAGES[0:100]]})

    def __prepare_validation_images__(self, validation_images):
        self.V: int = validation_images.shape[0]
        self.T_validation_images = torch.from_numpy(
            validation_images).type(torch.float)
        self.validation_sinograms = OperatorFunction.apply(self.radon, self.T_validation_images).data

        if self.DEBUG_PLOT:
            if self.V < 10:
                for i in range(self.V):
                    plot_imshow(
                        validation_images[i], title=f"Validation image - {i}")
            else:
                for i in range(10):
                    plot_imshow(
                        validation_images[i], title=f"Validation image - {i}")

        if self.USE_WANDB:
            if self.V < 10:
                wandb.log(
                    {"validation images": [wandb.Image(img) for img in validation_images]})
            else:
                wandb.log({"validation images": [wandb.Image(
                    img) for img in validation_images[0:100]]})

    def __prepare_angles__(self):
        self.angles_degrees = torch.linspace(0, 360, self.N).type(torch.float)
        self.angles_np = np.linspace(0, 2 * np.pi, self.N)
        self.__prepare_radon__()

    def __prepare_radon__(self):
         # Reconstruction space: discretized functions on the rectangle
        # [-20, 20]^2 with 300 samples per dimension.
        reco_space = odl.uniform_discr(
            min_pt=[-20, -20], max_pt=[20, 20], shape=[self.RESOLUTION, self.RESOLUTION], dtype='float32')

        # Angles: uniformly spaced, n = 1000, min = 0, max = pi
        angle_partition = odl.uniform_partition(0, np.pi, self.N)

        # Detector: uniformly sampled, n = 500, min = -30, max = 30
        detector_partition = odl.uniform_partition(-30, 30, self.RESOLUTION)

        # Make a parallel beam geometry with flat detector
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

        # --- Create Filtered Back-projection (FBP) operator --- #
        # Ray transform (= forward projection).
        self.radon = odl.tomo.RayTransform(reco_space, geometry)

        # Fourier transform in detector direction
        fourier = odl.trafos.FourierTransform(self.radon.range, axes=[1])
        # Create ramp in the detector direction
        ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
        # Create ramp filter via the convolution formula with fourier transforms
        ramp_filter = fourier.inverse * ramp_function * fourier

        # Create filtered back-projection by composing the back-projection (adjoint)
        # with the ramp filter.
        self.fbp = self.radon.adjoint * ramp_filter
    
    def __forward__(self):
        self.sinograms = OperatorFunction.apply(self.radon, self.T_input_images).data

    def __prepare_graph__(self):
        points_np = np.array(
            [np.cos(self.angles_np), np.sin(self.angles_np)]).T
        distances = haversine_distances(points_np, points_np)
        graph, classes, edges = generate_knn_from_distances_with_edges(
            distances, self.K, ordering='asc', ignoreFirst=True)
        N_noisy_edges = len(edges)
        self.graph_edges = np.zeros((2, N_noisy_edges))
        noisy_edges_list = list(edges)
        for i in range(N_noisy_edges):
            (n, m) = noisy_edges_list[i]
            self.graph_edges[0, i] = n
            self.graph_edges[1, i] = m

    def __prepare_model__(self):
        self.model_sinogram = GAT(
            in_dim=self.RESOLUTION,
            hidden_dim=self.RESOLUTION // self.GAT_HEADS,
            num_layers=self.GAT_LAYERS,
            out_dim=self.RESOLUTION,
            heads=self.GAT_HEADS,
            dropout=self.GAT_DROPOUT)

        self.model_sinogram = DataParallel(self.model_sinogram)
        self.model_sinogram.to(self.device)

        self.model_fbp = OperatorModule(self.fbp)

        self.optimizer_sinogram = torch.optim.Adam(self.model_sinogram.parameters(
        ), lr=self.GAT_ADAM_LR, weight_decay=self.GAT_ADAM_WEIGHTDECAY)

    def _prepare_data(self, batch_size):
        if self.FIXED_SNR:
            if self.loader is None:
                self.noisy_sinograms = _add_noise_to_sinograms(
                    self.sinograms, self.SNR)
                self.dataset = CustomSinogramDataset(
                    self.M, self.T_input_images, self.noisy_sinograms, torch.tensor(self.graph_edges).type(torch.long))
                self.loader = DataListLoader(
                    dataset=self.dataset, batch_size=batch_size, shuffle=True)
            return self.loader, self.SNR
        else:
            snr = torch.randint(self.SNR_LOWER, self.SNR_UPPER, (1,))
            noisy_sinograms = _add_noise_to_sinograms(self.sinograms, snr)
            dataset = CustomSinogramDataset(self.M, self.T_input_images, noisy_sinograms, torch.tensor(
                self.graph_edges).type(torch.long))
            loader = DataListLoader(
                dataset=dataset, batch_size=batch_size, shuffle=True)
            return loader, snr

    def _prepare_validation_data(self, snr, batch_size):
        validation_dataset = CustomValidationSinogramDataset(
            self.V, snr, self.validation_sinograms, self.T_validation_images, torch.tensor(self.graph_edges).type(torch.long))
        loader = DataListLoader(
            dataset=validation_dataset, batch_size=batch_size, shuffle=False)
        return loader

    def train(self, batch_size):
        if self.type != 'denoiser':
            raise RuntimeError("Only denoiser instance can be trained")

        self.__execute_and_log_time__(lambda: self.__train__(batch_size), "training")
        return self.model_sinogram

    def __train__(self, batch_size):
        self.model_sinogram.train()

        for epoch in range(self.EPOCHS):
            loader, snr = self._prepare_data(batch_size)
            loss_epoch = 0
            for data in loader:
                self.optimizer_sinogram.zero_grad()
                out_sinograms = self.model_sinogram(data)
                batch_n = int(out_sinograms.size(dim=0) / self.N)

                y_clean = torch.cat([d.y for d in data]).to(out_sinograms.device).view(batch_n, self.RESOLUTION, self.RESOLUTION)
                
                fbps = self.model_fbp(out_sinograms.view(batch_n, self.N, self.RESOLUTION))

                loss_fbps = torch.linalg.norm(fbps - y_clean)

                loss_fbps.backward()
                self.optimizer_sinogram.step()
                loss_epoch += loss_fbps

            print(
                f"epoch : {epoch} , epoch-snr: {snr}, epoch_loss : {loss_epoch}")
            if self.USE_WANDB:
                wandb.log({
                    "epoch": epoch,
                    "epoch_loss": loss_epoch,
                    "epoch_snr": snr
                })

        return self.model_sinogram

    def validate(self, validation_images, validation_snrs, batch_size):
        self.__execute_and_log_time__(lambda: self.__validate__(
            validation_images, validation_snrs, batch_size), "validation")

    def __validate__(self, validation_images, validation_snrs, batch_size):
        self.__prepare_validation_images__(validation_images)

        self.model_sinogram.eval()
        self.model_sinogram.to(self.device0)

        validation_loss_score = 0
        validation_loss_noisy_score = 0

        counter = 0

        for snr in validation_snrs:
            loader = iter(self._prepare_validation_data(snr, batch_size))

            validation_loss_per_snr = 0
            validation_loss_noisy_per_snr = 0

            for validation_data in loader:
                denoised_sinograms = self.model_sinogram(validation_data) # denoised sino 
                batch_n = int(denoised_sinograms.size(dim=0) / self.N)

                clean_images = torch.cat([d.y for d in validation_data]).to(denoised_sinograms.device).view(batch_n, self.RESOLUTION, self.RESOLUTION) # clean images
                noisy_sinograms = torch.cat([d.x for d in validation_data]).to(denoised_sinograms.device) # noisy sino
                
                fbps_denoised = self.model_fbp(denoised_sinograms.view(batch_n, self.N, self.RESOLUTION))
                fbps_noisy = self.model_fbp(noisy_sinograms.view(batch_n, self.N, self.RESOLUTION))
                

                validation_loss_per_snr += torch.linalg.norm(fbps_denoised - clean_images)
                validation_loss_noisy_per_snr += torch.linalg.norm(fbps_noisy - clean_images)

                for i in range(batch_n):
                    denoised_reconstruction = fbps_denoised[i]
                    noisy_reconstruction = fbps_noisy[i]

                    if self.USE_WANDB:
                        wandb.log({
                            "val_indx" : counter,
                            "val_loss": torch.linalg.norm(denoised_reconstruction.cpu().detach() - clean_images[i].cpu().detach()),
                            "val_loss_noisy": torch.linalg.norm(noisy_reconstruction.cpu().detach() - clean_images[i].cpu().detach()),
                            "val_input_snr_calculated": _find_SNR(clean_images[i].cpu().detach(), fbps_noisy[i].cpu().detach()),
                            "val_denoised_snr_calculated": _find_SNR(clean_images[i].cpu().detach(), fbps_denoised[i].cpu().detach()),
                            "val_denoised_reconstruction": wandb.Image(denoised_reconstruction.cpu().detach().numpy(), caption=f"Rec denoised - SNR: {snr} "),
                            "val_noisy reconstruction": wandb.Image(noisy_reconstruction.cpu().detach().numpy(), caption=f"Rec noisy - SNR: {snr}"),
                        })
                        counter += 1

            print(
                f"Validation loss snr {snr} : {validation_loss_per_snr} -- loss noisy: {validation_loss_noisy_per_snr}")
            validation_loss_score += validation_loss_per_snr
            validation_loss_noisy_score += validation_loss_noisy_per_snr
            
            if self.USE_WANDB:
                wandb.log({
                    "val_snr": snr,
                    "val_snr_loss_score": validation_loss_per_snr,
                    "val_snr_loss_noisy_score": validation_loss_noisy_per_snr})
            
            torch.cuda.empty_cache()

        if self.USE_WANDB:
            wandb.log({
                "val_loss_score": validation_loss_score,
                "val_loss_noisy_score": validation_loss_noisy_score})

    def finish_wandb(self, execution_time=None):
        if execution_time != None:
            wandb.log({"execution_time": execution_time})

        wandb.finish()

    def init_wandb(self, wandb_name, args, run_name = None):
        if self.USE_WANDB:
            if args is None:
                config = {
                    "samples": self.N,
                    "resolution": self.RESOLUTION,
                    "k-nn": self.K,
                    "gat-epochs": self.EPOCHS,
                    "gat-layers": self.GAT_LAYERS,
                    "gat-heads": self.GAT_HEADS,
                    "gat-DROPOUT": self.GAT_DROPOUT,
                    "gat-adam-weightdecay": self.GAT_ADAM_WEIGHTDECAY,
                    "gat-adam-learningrate": self.GAT_ADAM_LR,
                    "gat_snr_lower_bound": self.SNR_LOWER,
                    "gat_snr_upper_bound": self.SNR_UPPER,
                    "M": self.M,
                    # "name" : image_name
                }
            else:
                config = args

            gpus = torch.cuda.device_count()
            config.gpu_count = gpus

            wandb.init(project=wandb_name, entity="cedric-mendelin",
                       config=config, reinit=False)

            if self.type == 'denoiser':
                wandb.run.name = f"{wandb_name}-{gpus}-{self.RESOLUTION}-{self.N}-{self.M}-{self.K}-{self.EPOCHS}-{self.GAT_LAYERS}-{self.GAT_HEADS}-{self.GAT_DROPOUT}-{self.GAT_ADAM_WEIGHTDECAY}"
            elif run_name is not None :
                wandb.run.name = run_name

    def __execute_and_log_time__(self, action, name):
        if self.VERBOSE:
            t = time.time()

        action()

        if self.VERBOSE:
            self.time_dict[name] = time.time()-t
            print(f"{name} finished in :", self.time_dict[name])

            if self.USE_WANDB:
                wandb.log({f"{name}": self.time_dict[name]})
