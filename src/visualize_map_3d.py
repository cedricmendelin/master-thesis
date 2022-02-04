"""
This script enables to visualize a map file for debugging purpose.
The script uses vedo for visualization.
"""

from matplotlib.pyplot import plot
import mrcfile
import numpy as np
import logging
from utils.Data import *
from utils.Plotting import *
from utils.Vedo import visualize_voxels_3d
from aspire.volume import Volume
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

# Parameters:
#DATA_DIR = "src/maps/"  # Tutorial example data folder
DATA_DIR = 'src/reconstruction/'  # Tutorial example data folder
map_name = 'rec_given_rots_chengBunny.map'


# Check if parameters suitable 
normalize_data_min_max = True
cut_negatives_and_small_values = False
invert_normalization = True

should_downsample = False
downsample_size = 50


# Main:

v_npy = mrcfile.open( DATA_DIR + map_name ).data.astype(np.float64)
print(v_npy.shape)

if should_downsample:
  frac = downsample_size / v_npy.shape[0]
  v_npy = zoom(v_npy, (frac, frac, frac))
  #v_npy = Volume(v_npy).downsample(downsample_size).__getitem__(0)

if cut_negatives_and_small_values:
  v_npy = set_negatives_and_small_values_to(v_npy)

if normalize_data_min_max:
  v_npy = normalize_min_max(v_npy)

if invert_normalization:
  v_npy = 1 - v_npy



# v_npy = normalize_min_max(v_npy)


plot_imshow(v_npy[:,:,25])

#threshold = int(input("enter threshold:"))

threshold = 0.4

v_npy[v_npy < threshold] = 0



logger.info(f"Data max = {v_npy.max()}")
logger.info(f"Data min = {v_npy.min()}")
logger.info(f"Data mean = {v_npy.mean()}")
logger.info(f"Data shape = {v_npy.shape}")
logger.info(f"Data std = {v_npy.std()}")


#v_npy[v_npy < 0.03] = 0

visualize_voxels_3d(v_npy)


#v_npy[v_npy < 0.04] = 0

# plot_voxels(v_npy)

input("Enter to terminate")