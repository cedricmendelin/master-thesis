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

mesh = Mesh(dataurl + "bunny.obj").normalize().subdivide()
num_vols = 1  # number of volumes
img_size = 60  # image size in square
n_img = 100  # number of images
num_eigs = 16  # number of eigen-vectors to keep


#pts = mesh.points(copy=True)  # pts is a copy of the points not a reference

vol = mesh2Volume(mesh, spacing=(0.01,0.01,0.01))

#print(vol.tonumpy().shape)
print(vol.dimensions())


vol_pad = np.pad(vol.tonumpy(), ((0,3),(0,6),(0,57)), mode='constant', constant_values=0).astype(float)

print(vol_pad.dtype)
print (vol_pad.shape)


#vol_pad = np.pad(vol.tonumpy(), (237,237,237), constant_values=0)

# Then using that to instantiate a Volume, which is downsampled to 60x60x60
v = Volume(vol_pad)

v = v.downsample(img_size)


test = vedo.Volume(v.__getitem__(0))
test.show()

# %%
# Defining rotations
# ------------------
# This will force a collection of in plane rotations about z.

# First get a list of angles to test
thetas = np.linspace(0, 2 * np.pi, num=n_img, endpoint=False)

# Define helper function for common 3D rotation matrix, about z.


def r_z(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


# Construct a sequence of rotation matrices using r_z(thetas)
_rots = np.empty((n_img, 3, 3))
for n, theta in enumerate(thetas):
    # Note we negate theta to match Rotation convention.
    _rots[n] = r_z(-theta)

# Instantiate ASPIRE's Rotation class with the rotation matrices.
# This will allow us to use or access the rotations in a variety of ways.
rots = Rotation.from_matrix(_rots)

# %%
# Configure Noise
# ---------------
# We can define controlled noise and have the Simulation apply it to our projection images.

noise_variance = 1e-10  # Normally this would be derived from a desired SNR.

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
    vols=v,  # our Volume
    L=v.resolution,  # resolution, should match Volume
    n=n_img,  # number of projection images
    C=len(v),  # Number of volumes in vols. 1 in this case
    angles=rots.angles,  # pass our rotations as Euler angles
    offsets=shifts,  # translations (wrt to origin)
    amplitudes=amplitudes,  # amplification ( 1 is identity)
    seed=12345,  # RNG seed for reproducibility
    dtype=v.dtype,  # match our datatype to the Volume.
    noise_filter=white_noise_filter,  # optionally prescribe noise
)

# %%
# Yield projection images from the Simulation Source
# --------------------------------------------------

# Consume images from the source by providing
# a starting index and number of images.
# Here we generate the first 3 and peek at them.
src.images(0, 3).show()
src.projections(0, 3).show()

# Here we return the first n_img images as a numpy array.
dirty_ary = src.images(0, n_img).asnumpy()

# And we have access to the clean images
clean_ary = src.projections(0, n_img).asnumpy()



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

test = vedo.Volume(mean_est.__getitem__(0))
test.show()