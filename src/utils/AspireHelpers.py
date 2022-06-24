import numpy as np

from aspire.operators import ScalarFilter
from aspire.source.simulation import Simulation
from aspire.operators import RadialCTFFilter

def default_shifts(n_img):
  return  np.zeros((n_img, 2))

def default_amplitudes(n_img):
  return np.ones(n_img)


def create_simulation(aspire_vol, n, angles, noise_variance, shifts=None, amplitudes=None, ctf=None, seed=12345):
  """create aspire simultation


  Args:
      aspire_vol (aspire.Volume): 
      n (int): number of projections in sumulation
      angles (n, 3): euler angles for rotation
      noise_variance (float32): noise variance (sigma)
      shifts (, optional): [description]. Defaults to None.
      amplitudes ([type], optional): [description]. Defaults to None.
      ctf ([type], optional): [description]. Defaults to None.
      seed (int, optional): [description]. Defaults to 12345.

  Returns:
      [type]: [description]
  """
  if shifts is None:
    shifts = default_shifts(n)
  
  if amplitudes is None:
    amplitudes = default_amplitudes(n)

  if ctf is None:
    #ctf = np.linspace(1.5e4, 2.5e4, 7)
    ctf = np.zeros((7))

  white_noise_filter = ScalarFilter(dim=2, value=noise_variance)

  # Create a Simulation Source object
  src = Simulation(
      vols=aspire_vol,  # our Volume
      L=aspire_vol.resolution,  # resolution, should match Volume
      n=n,  # number of projection images
      C=len(aspire_vol),  # Number of volumes in vols. 1 in this case
      angles=angles,  # pass our rotations as Euler angles
      offsets=shifts,  # translations (wrt to origin)
      amplitudes=amplitudes,  # amplification ( 1 is identity)
      seed=seed,  # RNG seed for reproducibility
      dtype=aspire_vol.dtype,  # match our datatype to the Volume.
      noise_filter=white_noise_filter,  # optionally prescribe noise
      unique_filters= [RadialCTFFilter(defocus=d) for d in ctf]
  )

  return src


def get_clean_image_array(src, n):
  src.images(0, n).asnumpy()

def get_noisy_image_array(src, n):
  src.projections(0, n).asnumpy()