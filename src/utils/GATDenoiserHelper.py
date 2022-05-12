from abc import abstractmethod
from sklearn.metrics.pairwise import haversine_distances

from .ODLHelper import OperatorFunction, OperatorModule
from .ODLHelperCustom import setup_forward_and_backward

from .Graph import *
from .Plotting import *
from .ToyImageGenerator import *
from .PytorchGATHelper import *
from .SNRHelper import add_noise_to_sinograms, find_SNR

import numpy as np
import wandb
import time
from torch_geometric.nn import GATConv, DataParallel
from torch_geometric.loader import DataListLoader
from torch_geometric.data import Data, Dataset
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2

import odl


################### Loss Enum ##############################
from enum import Enum
class Loss(Enum):
    """ Enum class for different types of losses.
        Sino : l2-distance between clean sinogram and denoised sinogram.
        Fbp: l2.distance between clean input images and fbp(denoised sinogram).
    """
    SINO = 0,
    FBP = 1

################### Denoiser class #########################
class GatBase():
    """ Abstract base class for GatDenoisers and GatValidator
    """
    def __init__(self, args):
        self.consumeArgs(args)
        self.loader = None

        if self.VERBOSE:
            self.time_dict = {}

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.device0 = self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ######################## Initializer for child classes ############################
    def consumeArgs(self, args):
        self.args = args
        self.VERBOSE = args.verbose
        self.USE_WANDB = args.use_wandb

        self.N: int = args.graph_size
        self.K : int = args.k_nn
        
        self.RESOLUTION: int = args.resolution
        
        self.GAT_LAYERS: int = args.gat_layers
        self.GAT_HEADS: int = args.gat_heads
        self.GAT_DROPOUT: float = args.gat_dropout

    def consumeDenoiserArgs(self, args):
        self.M = args.samples
        self.EPOCHS: int = args.epochs
        
        self.GAT_ADAM_WEIGHTDECAY: float = args.gat_weight_decay
        self.GAT_ADAM_LR: float = args.gat_learning_rate
        self.Loss = Loss[args.loss.upper()]

        # check if snr is fixed
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

    @classmethod
    def create_validator(cls, args, model_state):
        """ Creates a validator instance.

        Args:
            args: Arguments for setting up the validator.
            model_state (dict): The trained model which will be validated.

        Returns:
            GatValidator: The created instance.
        """
        validator =  GatValidator(args)
        validator.__initialize_validator__(args, model_state)
        return validator

    @classmethod
    def create_toyimage_dynamic_denoiser(cls, args):
        """ Create a dynamic toyimage denoiser.
        During training, the denoiser will create uniformly created toy images for every epoch.

        Args:
            args: Arguments for setting up the denoiser.

        Returns:
            GatDenoiserToyImagesDynamic: The created instance.
        """
        denoiser = GatDenoiserToyImagesDynamic(args)
        denoiser.__initialize_denoiser__(args)
        
        return denoiser

    @classmethod
    def create_fixed_images_denoiser(cls, args):
        """ Create a fixed image denoiser. In every epoch, the same training images will be used.

        Args:
            args: Arguments for setting up the denoiser.

        Returns:
            GatDenoiserImagesFixed: The created instance.
        """
        denoiser =  GatDenoiserImagesFixed(args)
        denoiser.__initialize_denoiser__(args)
        return denoiser
    
    def __initialize_validator__(self, args, model_state):
        if self.USE_WANDB:
            self.__init_wandb__(args.wandb_project, args)

        self.__execute_and_log_time__(lambda: self.__init_graph_and_forward_backward(args), "init")
        self.__execute_and_log_time__(lambda: self.__prepare_model__(model_state), "prep model")

    def __initialize_denoiser__(self, args):
        self.consumeDenoiserArgs(args)

        if self.USE_WANDB:
            self.__init_wandb__(args.wandb_project, args)
        
        self.__execute_and_log_time__(lambda: self.__init_graph_and_forward_backward(args), "init")
        self.__execute_and_log_time__(lambda: self.__prepare_model__(), "prep model")
        self.__execute_and_log_time__(lambda: self.__prepare_optimizer__(), "prep optimizer")

    def __init_wandb__(self, wandb_name, args, wandb_user ="cedric-mendelin", run_name = None):
        config = args

        gpus = torch.cuda.device_count()
        config.gpu_count = gpus

        wandb.init(project=wandb_name, entity=wandb_user, config=config, reinit=False)

        if isinstance(self, GatDenoiserImagesFixed) or isinstance(self, GatDenoiserToyImagesDynamic):
            wandb.run.name = f"{wandb_name}-{gpus}-{self.RESOLUTION}-{self.N}-{self.M}-{self.K}-{self.EPOCHS}-{self.GAT_LAYERS}-{self.GAT_HEADS}-{self.GAT_DROPOUT}-{self.GAT_ADAM_WEIGHTDECAY}-{self.SNR_LOWER}-{self.SNR_UPPER}-{self.Loss}"
        elif run_name is not None :
            wandb.run.name = run_name

    ################## Graph - setup during initialization #############
    def __init_graph_and_forward_backward(self, args):
        # parameters for forward and backward operatior
        # Currently radon and filter_back_projection is used.
        self.radon, self.fbp = setup_forward_and_backward(self.RESOLUTION, self.N)
        
        # Parameters for seeting up graph.
        # Currently a circle graph with k neighoburs is used.
        self.graph = self.__prepare_graph__()

    def __prepare_graph__(self):
        # sample equally spaced points on unit-circle
        angles_np = np.linspace(0, 2 * np.pi, self.N)
        points_np = np.array([np.cos(angles_np), np.sin(angles_np)]).T

        # calculate distances of points on unit-circle
        distances = haversine_distances(points_np, points_np)

        # create the k-nn graph from distances
        _, _, edges = generate_knn_from_distances_with_edges(distances, self.K, ordering='asc', ignoreFirst=True)
        return torch.tensor(edges.T).type(torch.long) 

    ################## Forward operator Operators ######################
    def __forward__(self, images):
        """ Uses odl for forwarding images.

        Args:
            images: images, shape: (N, Resolution, Resolution)
        Returns:
            numpy.array: sinograms of input images.
        """
        return OperatorFunction.apply(self.radon, images).data

    ################### Model - setup during initialization ############
    def __prepare_model__(self, state=None):
        self.model = GAT(
            in_dim=self.RESOLUTION,
            hidden_dim=self.RESOLUTION // self.GAT_HEADS,
            num_layers=self.GAT_LAYERS,
            out_dim=self.RESOLUTION,
            heads=self.GAT_HEADS,
            dropout=self.GAT_DROPOUT)

        if state != None:
            self.model.load_state_dict(state)

        # model for pytorch.Geometric data parallel
        # this is needed to train on multiple GPUs
        self.model = DataParallel(self.model)
        self.model.to(self.device)

        # ODL operator module to apply model to multiply sinograms
        self.model_fbp = OperatorModule(self.fbp)

    def __prepare_optimizer__(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.GAT_ADAM_LR, 
            weight_decay=self.GAT_ADAM_WEIGHTDECAY)

    @abstractmethod
    def _prepare_training_data(self, batch_size, loss):
        pass

    @abstractmethod 
    def _prepare_training_images(self, images):
        pass

    def _prepare_validation_data(self, snr, batch_size):
        validation_dataset = ValidationDataset(self.V, snr, self.validation_sinograms, self.graph)
        loader = DataListLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
        return loader

    def __prepare_validation_images__(self, validation_images):
        self.V: int = validation_images.shape[0]
        self.T_validation_images = torch.from_numpy(validation_images).type(torch.float)
        self.validation_sinograms = self.__forward__(self.T_validation_images)

        if self.USE_WANDB:
            if self.V < 10:
                wandb.log(
                    {"validation images": [wandb.Image(img) for img in validation_images]})
            else:
                wandb.log({"validation images": [wandb.Image(
                    img) for img in validation_images[0:100]]})

    def train(self, images=None, batch_size=1, loss=Loss.SINO):
        self.__execute_and_log_time__(lambda: self.__train__(images, batch_size, loss), "training")
        return self.model

    def __train__(self, images, batch_size, loss):
        self._prepare_training_images(images)
        self.model.train()

        for epoch in range(self.EPOCHS):
            # get training data, if snr and images are fixed, will only be calculated once
            loader, snr = self._prepare_training_data(batch_size, loss)
            loss_epoch = 0

            for data in loader:
                self.optimizer.zero_grad()
                out_sinograms = self.model(data)
                batch_n = int(out_sinograms.size(dim=0) / self.N)

                # if loss : sino 
                # y = clean_sinograms
                # if loss : fbp 
                # y = clean input images
                y = torch.cat([d.y for d in data]).to(out_sinograms.device)
                
                if loss == Loss.SINO:
                    loss_training = torch.linalg.norm(out_sinograms - y)
                elif loss == Loss.FBP:
                    fbps = self.model_fbp(out_sinograms.view(batch_n, self.N, self.RESOLUTION))
                    loss_training = torch.linalg(fbps.view(batch_n, self.RESOLUTION, self.RESOLUTION) - y.view(batch_n, self.RESOLUTION, self.RESOLUTION))
                else:
                    raise RuntimeError("Unknown loss type!")

                loss_training.backward()
                self.optimizer.step()
                loss_epoch += loss_training

            print(f"epoch : {epoch} , epoch-snr: {snr}, epoch_loss : {loss_epoch}")
            if self.USE_WANDB:
                wandb.log({
                    "epoch": epoch,
                    "epoch_loss": loss_epoch,
                    "epoch_snr": snr
                })

        return self.model


    def validate(self, validation_images, validation_snrs, batch_size):
        self.__execute_and_log_time__(lambda: self.__validate__(
            validation_images, validation_snrs, batch_size), "validation")

    def __validate__(self, validation_images, validation_snrs, batch_size):
        self.__prepare_validation_images__(validation_images)

        self.model.eval()
        self.model.to(self.device0)

        validation_loss_sino_denoised_score = 0
        validation_loss_sino_noisy_score = 0
        validation_loss_fbp_denoised_score = 0
        validation_loss_fbp_noisy_score = 0

        counter = 0
        image_index = 0

        for snr in validation_snrs:
            loader = iter(self._prepare_validation_data(snr, batch_size))

            validation_loss_fbp_denoised_per_snr = 0
            validation_loss_fbp_noisy_per_snr = 0
            validation_loss_sino_denoised_per_snr = 0
            validation_loss_sino_noisy_per_snr = 0

            for validation_data in loader:
                denoised_sinograms = self.model(validation_data) # denoised sino 
                batch_n = int(denoised_sinograms.size(dim=0) / self.N)

                denoised_sinograms = denoised_sinograms.view(batch_n, self.N, self.RESOLUTION)
                noisy_sinograms = torch.cat([d.x for d in validation_data]).view(batch_n, self.N, self.RESOLUTION).to(denoised_sinograms.device) # noisy sino
                clean_sinograms = torch.cat([d.y for d in validation_data]).view(batch_n, self.N, self.RESOLUTION).to(denoised_sinograms.device)
                
                fbps_denoised = self.model_fbp(denoised_sinograms).to(denoised_sinograms.device)
                fbps_noisy = self.model_fbp(noisy_sinograms).to(denoised_sinograms.device)
                
                clean_images = self.T_validation_images[image_index: image_index + batch_n,:,:].to(denoised_sinograms.device)
                image_index = image_index + batch_n

                for i in range(batch_n):
                    if self.USE_WANDB:
                        wandb.log({
                            "val_indx" : counter,
                            "val_loss_sino_denoised": torch.linalg.norm(denoised_sinograms[i]- clean_sinograms[i]),
                            "val_loss_sino_noisy": torch.linalg.norm(noisy_sinograms[i] - clean_sinograms[i]),

                            "val_loss_fbp_denoised": torch.linalg.norm(fbps_denoised[i] - clean_images[i]),
                            "val_loss_fbp_noisy": torch.linalg.norm(fbps_noisy[i] - clean_images[i]),

                            "val_snr_sino_denoised": find_SNR(clean_sinograms[i], denoised_sinograms[i]),
                            "val_snr_sino_noisy": find_SNR(clean_sinograms[i], noisy_sinograms[i]),

                            "val_snr_fbp_denoised": find_SNR(clean_images[i], fbps_denoised[i]),
                            "val_snr_fbp_noisy": find_SNR(clean_images[i], fbps_noisy[i]),

                            "val_reconstruction_denoised": wandb.Image(fbps_denoised[i].cpu().detach().numpy(), caption=f"Rec denoised - SNR: {snr} "),
                            "val_reconstruction_noisy": wandb.Image(fbps_noisy[i].cpu().detach().numpy(), caption=f"Rec noisy - SNR: {snr}"),

                            "val_sino_denoised": wandb.Image(denoised_sinograms[i].cpu().detach().numpy(), caption=f"Sinogram denoised - SNR: {snr} "),
                            "val_sino_noisy": wandb.Image(noisy_sinograms[i].cpu().detach().numpy(), caption=f"Sinogram noisy - SNR: {snr}"),
                            "val_sino_clean": wandb.Image(clean_sinograms[i].cpu().detach().numpy(), caption=f"Sinogram noisy - SNR: {snr}"),
                        })
                        counter += 1
            
                validation_loss_fbp_denoised_per_snr += torch.linalg.norm(fbps_denoised - clean_images)
                validation_loss_fbp_noisy_per_snr += torch.linalg.norm(fbps_noisy - clean_images)

                validation_loss_sino_denoised_per_snr += torch.linalg.norm(denoised_sinograms - clean_sinograms)
                validation_loss_sino_noisy_per_snr += torch.linalg.norm(noisy_sinograms - clean_sinograms)

            if self.USE_WANDB:
                wandb.log({
                    "val_snr": snr,
                    "val_loss_sino_denoised_snr": validation_loss_sino_denoised_per_snr,
                    "val_loss_sino_noisy_snr": validation_loss_sino_noisy_per_snr,
                    "val_loss_fbp_denoised_snr": validation_loss_fbp_denoised_per_snr,
                    "val_loss_fbp_noisy_snr": validation_loss_fbp_noisy_per_snr,})

            print(f"Validation loss sino snr {snr} : {validation_loss_sino_denoised_per_snr} -- loss noisy: {validation_loss_sino_noisy_per_snr}")
            print(f"Validation loss fbp snr {snr} : {validation_loss_fbp_denoised_per_snr} -- loss noisy: {validation_loss_fbp_noisy_per_snr}")
            
            validation_loss_sino_denoised_score += validation_loss_sino_denoised_per_snr
            validation_loss_sino_noisy_score += validation_loss_sino_noisy_per_snr
            validation_loss_fbp_denoised_score += validation_loss_fbp_denoised_per_snr
            validation_loss_fbp_noisy_score += validation_loss_fbp_noisy_per_snr
            
            image_index = 0
            torch.cuda.empty_cache()

        if self.USE_WANDB:
              wandb.log({
                    "val_snr": snr,
                    "val_loss_sino_denoised_total": validation_loss_sino_denoised_score,
                    "val_loss_sino_noisy_total": validation_loss_sino_noisy_score,
                    "val_loss_fbp_denoised_total": validation_loss_fbp_denoised_score,
                    "val_loss_fbp_noisy_total": validation_loss_fbp_noisy_score,})

    def finish_wandb(self, execution_time=None):
        if execution_time != None:
            wandb.log({"execution_time": execution_time})

        wandb.finish()

   

    def __execute_and_log_time__(self, action, name):
        if self.VERBOSE:
            t = time.time()

        action()

        if self.VERBOSE:
            self.time_dict[name] = time.time()-t
            print(f"{name} finished in :", self.time_dict[name])

            if self.USE_WANDB:
                wandb.log({f"{name}": self.time_dict[name]})

########################### Implementations ############################


class GatDenoiserToyImagesDynamic(GatBase):
    """ Special toy images denoiser. 
        This denoiser will generate new images in every epoch.
    """
    def _prepare_training_images(self, images):
        return

    def _prepare_training_data(self, batch_size, loss):
        images = draw_uniform_toyimages(self.RESOLUTION, 13, self.M)
        sinos = self.__forward__(torch.tensor(images).type(torch.float))

        if self.FIXED_SNR:
            snr = self.SNR
        else:
            snr = torch.randint(self.SNR_LOWER, self.SNR_UPPER, (1,))
        
        noisy_sinos = add_noise_to_sinograms(sinos, snr)

        if loss == Loss.SINO:
            dataset = SinogramToSinogramDataset(self.M, sinos, noisy_sinos, self.graph)
        elif loss == Loss.FBP:
            dataset = SinogramToImageDataset(self.M, images, noisy_sinos, self.graph)
        else:
            raise RuntimeError("Unknown loss value!")
        
        loader = DataListLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        return loader, snr

class GatDenoiserImagesFixed(GatBase):
    """ GatDenoiser for fixed images during training.
        During training, in every epoch the same images are passed to the model.
        Clean sinograms need to be only calculated once, if SNR is fixed, 
        noisy sinograms are calculated only once as well.
    """
    def _prepare_training_images(self, images):
        self.training_images = torch.tensor(images).type(torch.float)
        self.training_sinos = self.__forward__(self.training_images)
        self.M: int = images.shape[0]

    def _prepare_training_data(self, batch_size, loss):
        if self.FIXED_SNR:
            if self.loader is None:
                self.training_sinos_noisy = add_noise_to_sinograms(self.training_sinos, self.SNR)
            
                if loss == Loss.SINO:
                    self.dataset = SinogramToSinogramDataset(self.M, self.training_sinos, self.training_sinos_noisy, self.graph)
                elif loss == Loss.FBP:
                    self.dataset = SinogramToImageDataset(self.M, self.training_images, self.training_sinos_noisy, self.graph)
                else:
                    raise RuntimeError("Unknown loss value!")

                self.loader = DataListLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True)

            return self.loader, self.SNR
        else:
            snr = torch.randint(self.SNR_LOWER, self.SNR_UPPER, (1,))
            noisy_sinograms = add_noise_to_sinograms(self.training_sinos, snr)
            if loss == Loss.SINO:
                dataset = SinogramToSinogramDataset(self.M, self.training_sinos, noisy_sinograms, self.graph)
            elif loss == Loss.FBP:
                dataset = SinogramToImageDataset(self.M, self.training_images, noisy_sinograms, self.graph)
            else:
                raise RuntimeError("Unknown loss value!")

            loader = DataListLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
            return loader, snr

class GatValidator(GatBase):
    """ GatValidator class which is only able to run validator from given trained model.
        Validator can be logged to wandb as well.
    """
    def _prepare_training_images(self, images):
         raise RuntimeError("Validator cannot be trained")

    def _prepare_training_data(self, batch_size, images=None):
         raise RuntimeError("Validator cannot be trained")
