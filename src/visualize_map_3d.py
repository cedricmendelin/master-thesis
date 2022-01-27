"""
This script enables to visualize a map file for debugging purpose.
The script uses vedo for visualization.
"""

import mrcfile
import numpy as np
import logging
from utils.Data import *
from utils.Vedo import visualize_voxels_3d
from aspire.volume import Volume

logger = logging.getLogger(__name__)

# Parameters:
DATA_DIR = "src/maps/"  # Tutorial example data folder
map_name = 'rec_given_rots.map'
normalize_data_min_max = False
cut_negatives_and_small_values = True
should_downsample = False
downsample_size = 60

# Main:

v_npy = mrcfile.open(DATA_DIR + map_name ).data.astype(np.float32)

if cut_negatives_and_small_values:
  v_npy = set_negatives_and_small_values_to(v_npy)

if normalize_data_min_max:
  v_npy = normalize_min_max(v_npy)

if should_downsample:
  v_npy = Volume(v_npy).downsample(downsample_size).__getitem__(0)


logger.info(f"Data max = {v_npy.max()}")
logger.info(f"Data min = {v_npy.min()}")
logger.info(f"Data mean = {v_npy.mean()}")
logger.info(f"Data shape = {v_npy.shape}")


visualize_voxels_3d(v_npy)

input("Enter to terminate")