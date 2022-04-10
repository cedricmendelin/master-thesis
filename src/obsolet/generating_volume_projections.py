"""
Generating 3D Volume Projections
================================

This script illustrates using ASPIRE's Simulation source to
generate projections of a Volume using prescribed rotations.

"""

import logging
import os

import mrcfile
import numpy as np

import aspire

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

from utils.obsolete.Data import *

logger = logging.getLogger(__name__)

# %%
# Configure how many images we'd like to project
# ----------------------------------------------
# Specify parameters
# num_vols = 2  # number of volumes
img_size = 50  # image size in square
n_img = 1000  # number of images
num_eigs = 16  # number of eigen-vectors to keep

# %%
# Load our Volume data
# --------------------
# This example starts with an mrc, loading it as an numpy array

DATA_DIR = "src/maps"  # Tutorial example data folder
v_npy = mrcfile.open("C:\master-thesis\src\maps\\bunny.map").data.astype(np.float64)

v_npy = normalize_min_max(v_npy)


# v_npy = np.nan_to_num(v_npy)

# Then using that to instantiate a Volume, which is downsampled to 60x60x60
v = Volume(v_npy).downsample(img_size)

print(v.__getitem__(0).dtype)

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
    unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)]
)


print(src.get_metadata("__filter_indices"))


# %%
# Yield projection images from the Simulation Source
# --------------------------------------------------

# Consume images from the source by providing
# a starting index and number of images.
# Here we generate the first 3 and peek at them.
src.images(0, 3).show()
src.projections(0, 3).show()

# Here we return the first n_img images as a numpy array.
# dirty_ary = src.images(0, n_img).asnumpy()

# And we have access to the clean images
# clean_ary = src.clean_images(0, n_img).asnumpy()


# Similary, the angles/rotations/shifts/amplitudes etc.

############## Reconstruction ###############

sim = src
num_imgs = n_img

# Specify the normal FB basis method for expending the 2D images
basis = FBBasis3D((img_size, img_size, img_size))

logger.info("Basis done")

# Estimate the noise variance. This is needed for the covariance estimation step below.
noise_estimator = WhiteNoiseEstimator(sim, batchSize=500)
noise_variance_estimated = noise_estimator.estimate()
logger.info(f"Noise Variance = {noise_variance_estimated}")

# %%
# Estimate Mean Volume and Covariance
# -----------------------------------
#
# Estimate the mean. This uses conjugate gradient on the normal equations for
# the least-squares estimator of the mean volume. The mean volume is represented internally
# using the basis object, but the output is in the form of an
# L-by-L-by-L array.

mean_estimator = MeanEstimator(sim, basis)
mean_est = mean_estimator.estimate(tol=0.8)
logger.info("Mean estimator done")
voxelSaveAsMap(mean_est.__getitem__(0), 'mean_estimated_vol.map')

# Passing in a mean_kernel argument to the following constructor speeds up some calculations
covar_estimator = CovarianceEstimator(sim, basis, mean_kernel=mean_estimator.kernel)
covar_est = covar_estimator.estimate(mean_est, noise_variance, tol=0.95)
logger.info("Covariance estimator done")
voxelSaveAsMap(mean_est.__getitem__(0), 'covariance_estimated_vol.map')

# %%
# Use Top Eigenpairs to Form a Basis
# ----------------------------------

# Extract the top eigenvectors and eigenvalues of the covariance estimate.
# Since we know the population covariance is low-rank, we are only interested
# in the top eigenvectors.

eigs_est, lambdas_est = eigs(covar_est, num_eigs)

# Eigs returns column-major, so we transpose and construct a volume.
eigs_est = Volume(np.transpose(eigs_est, (3, 0, 1, 2)))

# Truncate the eigendecomposition. Since we know the true rank of the
# covariance matrix, we enforce it here.

eigs_est_trunc = Volume(eigs_est[: 1])  # hrmm not very convenient
lambdas_est_trunc = lambdas_est[: 1, : 1]


voxelSaveAsMap(mean_est.__getitem__(0), 'eigs_est_trunc.map')

# Estimate the coordinates in the eigenbasis. Given the images, we find the
# coordinates in the basis that minimize the mean squared error, given the
# (estimated) covariances of the volumes and the noise process.
coords_est = src_wiener_coords(
    sim, mean_est, eigs_est_trunc, lambdas_est_trunc, noise_variance
)

# Cluster the coordinates using k-means. Again, we know how many volumes
# we expect, so we can use this parameter here. Typically, one would take
# the number of clusters to be one plus the number of eigenvectors extracted.

# Since kmeans2 relies on randomness for initialization, important to push random seed to context manager here.
with Random(0):
    centers, vol_idx = kmeans2(coords_est.T, 1)
    centers = centers.squeeze()

# %%
# Performance Evaluation
# ----------------------

# Evaluate performance of mean estimation.

mean_perf = sim.eval_mean(mean_est)


# Evaluate performance of covariance estimation. We also evaluate the truncated
# eigendecomposition. This is expected to be a closer approximation since it
# imposes an additional low-rank condition on the estimate.

covar_perf = sim.eval_covar(covar_est)
eigs_perf = sim.eval_eigs(eigs_est_trunc, lambdas_est_trunc)

# Evaluate clustering performance.

clustering_accuracy = sim.eval_clustering(vol_idx)

# Assign the cluster centroids to the different images. Since we expect a discrete distribution of volumes
# (and therefore of coordinates), we assign the centroid coordinate to each image that belongs to that cluster.
# Evaluate the coordinates estimated

clustered_coords_est = centers[vol_idx]
coords_perf = sim.eval_coords(mean_est, eigs_est_trunc, clustered_coords_est)

# %%
# Results
# -------

# Output estimated covariance spectrum.

logger.info(f"Population Covariance Spectrum = {np.diag(lambdas_est)}")


# Output performance results.

logger.info(f'Mean (rel. error) = {mean_perf["rel_err"]}')
logger.info(f'Mean (correlation) = {mean_perf["corr"]}')
logger.info(f'Covariance (rel. error) = {covar_perf["rel_err"]}')
logger.info(f'Covariance (correlation) = {covar_perf["corr"]}')
logger.info(f'Eigendecomposition (rel. error) = {eigs_perf["rel_err"]}')
logger.info(f"Clustering (accuracy) = {clustering_accuracy}")
logger.info(f'Coordinates (mean rel. error) = {coords_perf["rel_err"]}')
logger.info(f'Coordinates (mean correlation) = {np.mean(coords_perf["corr"])}')

# Basic Check
assert covar_perf["rel_err"] <= 0.60
assert np.mean(coords_perf["corr"]) >= 0.98
assert clustering_accuracy >= 0.99