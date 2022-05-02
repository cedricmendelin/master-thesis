import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from skimage.data import shepp_logan_phantom
import torch
import numpy as np
from utils.pytorch_radon.radon import *
from skimage.transform import  rescale
from utils.Plotting import *
from utils.ODLHelper import *


import scipy.stats as stats

input = shepp_logan_phantom()
print (input.shape)

scaleX = 200 / input.shape[0]
scaleY = 200 / input.shape[1]
input = rescale(input, scale=(scaleX, scaleY), mode='reflect', multichannel=False)

N = 1024
RESOLUTION = 200

angles_degrees =  torch.linspace(0, 360, N).type(torch.float)


################### Forward ##############################
radon_class = Radon(RESOLUTION, angles_degrees, circle=True)
 
input_t = torch.from_numpy(input).type(torch.float)

sinogram = radon_class.forward(input_t.view(1, 1, RESOLUTION, RESOLUTION))[0,0].T

op_64_32 = ParallelBeamGeometryOp(N, RESOLUTION)
# phantom = odl.phantom.shepp_logan(
#   op_64_32.reco_space, modified=True)
 
#x = torch.from_numpy(phantom.data)
y = op_64_32(input_t)

plot_imshow(y)


plot_imshow(input_t)

plot_imshow(sinogram)

rec = filterBackprojection2D(sinogram, angles_degrees)


sinogram2 = radon_class.forward(rec.view(1, 1, RESOLUTION, RESOLUTION))[0,0].T

rec2 = filterBackprojection2D(sinogram2, angles_degrees)

plot_imshow(rec)
plot_imshow(rec2)


zscores = stats.zscore(rec)

zscore_min = np.min(zscores)
zscore_max = np.max(zscores)

z_normalized = (zscores - zscore_min) / (zscore_max - zscore_min)

plot_imshow(z_normalized)

plt.show()