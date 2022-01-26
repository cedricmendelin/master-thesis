from aspire.storage.starfile import StarFile
from aspire.source.relion import RelionSource
from aspire.basis import FBBasis3D
from aspire.noise import WhiteNoiseEstimator
from aspire.reconstruction import MeanEstimator
from aspire.source.simulation import Simulation

import numpy as np

# Parameters:
img_size = 40


star = StarFile('out.star')

print(star)

relion = RelionSource('out.star')


relion.images(0,3).show()



#relion.
#relion.projections(0, 3).show()

print(relion.angles)


basis = FBBasis3D((img_size, img_size, img_size))

# Estimate the noise variance. This is needed for the covariance estimation step below.
noise_estimator = WhiteNoiseEstimator(relion, batchSize=500)
noise_variance = noise_estimator.estimate()

# %%
# Estimate Mean Volume and Covariance
# -----------------------------------
#
# Estimate the mean. This uses conjugate gradient on the normal equations for
# the least-squares estimator of the mean volume. The mean volume is represented internally
# using the basis object, but the output is in the form of an
# L-by-L-by-L array.

mean_estimator = MeanEstimator(relion, basis)
mean_est = mean_estimator.estimate()


input("Enter to terminate")