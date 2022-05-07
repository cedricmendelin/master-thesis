import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as torch_fft
from .utils import PI, SQRT2, deg2rad, affine_grid, grid_sample
from .filters import RampFilter

class Radon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True, dtype=torch.float):
        super(Radon, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.all_grids = None
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x):
        N, C, W, H = x.shape
        assert W == H

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape
        out = torch.zeros(N, C, W, len(self.theta), device=x.device, dtype=self.dtype)

        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            out[..., i] = rotated.sum(2)

        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2*grid_size).ceil())
        grid_shape = [1, 1, grid_size, grid_size]
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta, self.dtype)
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)
            all_grids.append(affine_grid(R, grid_shape))
        return all_grids

class IRadon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True,
                 use_filter=RampFilter(), out_size=None, dtype=torch.float):
        super(IRadon, self).__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype
        self.deg2rad = lambda x: deg2rad(x, dtype)
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle)
            self.all_grids = self._create_grids(self.theta, in_size, circle)
        self.filter = use_filter if use_filter is not None else lambda x: x

    def forward(self, x):
        it_size = x.shape[2]
        ch_size = x.shape[1]

        if self.in_size is None:
            self.in_size = int((it_size/SQRT2).floor()) if not self.circle else it_size
        if None in [self.ygrid, self.xgrid, self.all_grids]:
            self.ygrid, self.xgrid = self._create_yxgrid(self.in_size, self.circle)
            self.all_grids = self._create_grids(self.theta, self.in_size, self.circle)

        x = self.filter(x)

        reco = torch.zeros(x.shape[0], ch_size, it_size, it_size, device=x.device, dtype=self.dtype)
        for i_theta in range(len(self.theta)):
            reco += grid_sample(x, self.all_grids[i_theta].repeat(reco.shape[0], 1, 1, 1).to(x.device))

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1]))

        if self.circle:
            reconstruction_circle = (self.xgrid ** 2 + self.ygrid ** 2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(x.shape[0], ch_size, 1, 1)
            reco[~reconstruction_circle] = 0.

        reco *= PI.to(reco.device)/(2*len(self.theta))

        if self.out_size is not None:
            pad = (self.out_size - self.in_size)//2
            reco = F.pad(reco, (pad, pad, pad, pad))

        return reco

    def _create_yxgrid(self, in_size, circle):
        if not circle:
            in_size = int((SQRT2*in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype)
        return torch.meshgrid(unitrange, unitrange)

    def _xy_to_t(self, theta):
        return self.xgrid*self.deg2rad(theta).cos() - self.ygrid*self.deg2rad(theta).sin()

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2*grid_size).ceil())
        all_grids = []
        for i_theta, theta in enumerate(angles):
            X = torch.ones([grid_size]*2, dtype=self.dtype)*i_theta*2./(len(angles)-1)-1.
            Y = self._xy_to_t(theta)
            all_grids.append(torch.stack((X, Y), dim=-1).unsqueeze(0))
        return all_grids



def backprojection2D(sinogram,theta):
    #sinogram: SAMPLES x dimension
    #theta: SAMPLES in radon
    samples = sinogram.shape[0]
    dim = sinogram.shape[1]
    xgrid,ygrid = create_yxgrid(dim,torch.float32)
    xgrid = xgrid.to(sinogram.device)
    ygrid = ygrid.to(sinogram.device)
    
    circle = xgrid**2 + ygrid**2 < 1 
    
    BPReconstruction = torch.zeros((dim,dim)).to(sinogram.device)
    #print(sinogram.device)
    #print(theta.device)
    for theta, sino in zip(theta,sinogram):
        sinoExpand =  sino.repeat(sino.shape[0],1)
        sinoRotate =  rotateImage(sinoExpand,theta)
        BPReconstruction = BPReconstruction + sinoRotate*circle
        
    return BPReconstruction
        
def filterBackprojection2D(sinogram,theta):
    #sinogram: SAMPLES x dimension
    #theta: SAMPLES in radon 


    samples = sinogram.shape[0]
    dim = sinogram.shape[1]
    
    impulseProjection = torch.zeros_like(sinogram).to(sinogram.device)
    impulseProjection[:,int(dim/2)] = 1
    
    
    #FilteringPart
    #print(sinogram.device)
    sinogramFilter = filterSignal(sinogram).type(torch.FloatTensor).to(sinogram.device)
    #print(sinogramFilter.device)
    sinBP = backprojection2D(sinogramFilter,theta)
    #impulseBP = backprojection2D(impulseProjection,theta)
    
    #sinBPFFT  = torch_fft.fftn(sinBP)
    #impulseBPFFT = torch_fft.fftn(impulseBP)
    
    #filteredImage = torch.roll(torch.fft.ifftn(sinBPFFT/impulseBPFFT),(int(dim/2),int(dim/2)), dims=(1,0))
    
    return sinBP#filteredImage.real, impulseBP

def filterSignal(signal):
    #signal of the form N x dim
    #N : number of signals
    #print(signal.device)
    signalFFT = torch_fft.fft(signal)
    #print(signalFFT.device)
    filterCoef = ramp_filter(signal.shape[1]).real.to(signal.device)
    #print(filterCoef.device)
    return torch_fft.ifft(signalFFT*filterCoef).real
    
def ramp_filter(size):
    image_n = torch.cat([
        torch.arange(1, size / 2 + 1, 2, dtype=torch.int),
        torch.arange(size / 2 - 1, 0, -2, dtype=torch.int),
    ])
    
    
    #print(image_n.shape)

    image_filter = torch.zeros(size, dtype=torch.double)
    image_filter[0] = 0.25
    image_filter[1::2] = -1 / (PI * image_n) ** 2
    fourier_filter = torch_fft.fft(image_filter)
    #fourier_filter = torch.rfft(image_filter, 1, onesided=False)
    #fourier_filter[:, 1] = fourier_filter[:, 0]
    return 2*fourier_filter#,image_filter


def create_yxgrid(in_size,dtype):
    unitrange = torch.linspace(-1, 1, in_size, dtype=dtype)
    return torch.meshgrid(unitrange, unitrange)


def rotateImage(image, theta):
    image = image.unsqueeze(0).unsqueeze(0)


    piValue = 2*torch.arccos(torch.FloatTensor([0])).to(image.device)
    angle_rad = -theta *piValue/180
    rotation = torch.tensor([
    [ torch.cos(angle_rad), torch.sin(angle_rad), 0],
    [-torch.sin(angle_rad), torch.cos(angle_rad), 0],]).to(image.device)
    grid = F.affine_grid(rotation.unsqueeze(0), size=image.size(),align_corners=False)
    rotated = F.grid_sample(image, grid)
    return rotated[0,0]
