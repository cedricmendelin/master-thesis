import numpy as np
from vedo import Mesh, dataurl, mesh2Volume
import os.path as path
import vedo

from aspire.operators import ScalarFilter
from aspire.source.simulation import Simulation
from aspire.utils import Rotation
from aspire.volume import Volume
from scipy.cluster.vq import kmeans2

from aspire.basis import FBBasis3D
from aspire.covariance import CovarianceEstimator
from aspire.denoising import src_wiener_coords
from aspire.noise import WhiteNoiseEstimator
from aspire.operators import RadialCTFFilter
from aspire.reconstruction import MeanEstimator
from aspire.source.simulation import Simulation
from aspire.utils import eigs
from aspire.utils.random import Random
from aspire.volume import Volume

from utils.Data import *



aspire_vol = vedo_bunny_to_asipre_volume()
vedo_vol = vedo.Volume(aspire_vol.__getitem__(0))

vedo_vol.show()


num_vols = 1  # number of volumes
img_size = 40  # image size in square
n_img = 1000  # number of images
snr = 10
#num_eigs = 16  # number of eigen-vectors to keep


## prepare simulation

aspire_vol = aspire_vol.downsample(img_size)


# %%
# Defining rotations

_rots = random_rotation_3d(n_img)

# Instantiate ASPIRE's Rotation class with the rotation matrices.
# This will allow us to use or access the rotations in a variety of ways.
aspire_rots = Rotation.from_euler(_rots)


# %%
# Configure Noise
# ---------------
# We can define controlled noise and have the Simulation apply it to our projection images.
# Normally this would be derived from a desired SNR.
noise_variance = find_sigma_noise(snr, aspire_vol)

# Then create a constant filter based on that variance, which is passed to Simulation
white_noise_filter = ScalarFilter(dim=2, value=noise_variance)


# %%
# Setup Simulation Source
# -----------------------

# Simulation will randomly shift and amplify images by default.
# Instead we define the following parameters.
shifts = np.zeros((n_img, 2))
amplitudes = np.ones(n_img)

# Create a Simulation Source object
src = Simulation(
    vols=aspire_vol,  # our Volume
    L=aspire_vol.resolution,  # resolution, should match Volume
    n=n_img,  # number of projection images
    C=len(aspire_vol),  # Number of volumes in vols. 1 in this case
    angles=aspire_rots.angles,  # pass our rotations as Euler angles
    offsets=shifts,  # translations (wrt to origin)
    amplitudes=amplitudes,  # amplification ( 1 is identity)
    seed=12345,  # RNG seed for reproducibility
    dtype=aspire_vol.dtype,  # match our datatype to the Volume.
    noise_filter=white_noise_filter,  # optionally prescribe noise
    unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
)

# %%
# Yield projection images from the Simulation Source
# --------------------------------------------------

# Consume images from the source by providing
# a starting index and number of images.
# Here we generate the first 3 and peek at them.
#src.images(0, 3).show()
#src.projections(0, 3).show()


# Here we return the first n_img images as a numpy array.

dirty_ary = src.images(0, n_img).asnumpy()

#t = src.get_metadata("__filter_indices")
#print(aspire_vol.dtype)
#t = dirty_ary[0].get_metadata("__filter_indices")
#print(t)


# And we have access to the clean images
clean_ary = src.projections(0, n_img).asnumpy()

dirty_ary.shape



#### Reconstruction #####


basis = FBBasis3D((img_size, img_size, img_size))

# Estimate the noise variance. This is needed for the covariance estimation step below.
noise_estimator = WhiteNoiseEstimator(src, batchSize=500)
noise_variance = noise_estimator.estimate()

# %%
# Estimate Mean Volume and Covariance
# -----------------------------------
#
# Estimate the mean. This uses conjugate gradient on the normal equations for
# the least-squares estimator of the mean volume. The mean volume is represented internally
# using the basis object, but the output is in the form of an
# L-by-L-by-L array.

mean_estimator = MeanEstimator(src, basis)
mean_est = mean_estimator.estimate()

print("estimation finished")
voxelSaveAsMap(mean_est.__getitem__(0))

vedo_mean_est = aspire_volume_to_vedo_volume(mean_est)
print(mean_est.__getitem__(0).shape)

v = vedo.Volume(mean_est.__getitem__(0))
v.show()

print("transfer finished")

vedo_mean_est.show()

print("show")

print(mean_est)



input("Enter to terminate")



