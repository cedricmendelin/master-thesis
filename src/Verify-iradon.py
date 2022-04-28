from skimage.data import shepp_logan_phantom
import torch
import numpy as np
from utils.pytorch_radon.radon import *
from skimage.transform import  rescale
from utils.Plotting import *

input = shepp_logan_phantom()
input = rescale(input, input.size[0] / 200, input.size[1]/200)

N = 1024
RESOLUTION = 200

angles_degrees =  torch.linspace(0, 360, N).type(torch.float)


################### Forward ##############################
radon_class = Radon(RESOLUTION, angles_degrees, circle=True)
 
input_t = torch.from_numpy(input).type(torch.float)

sinogram = radon_class.forward(input_t.view(1, 1, RESOLUTION, RESOLUTION))


plot_imshow(input_t)

plot_imshow(sinogram)

plt.show()