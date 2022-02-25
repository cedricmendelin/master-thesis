import dis
import numpy as np
from .Geometry import random_rotation_3d
from .Normalization import *
from .Graph import *
from .AspireHelpers import *
import os.path as path
from os import mkdir
import gemmi
import vedo
import mrcfile

from aspire.volume import Volume
DATA_DIR = 'src/data/'
MAP_DIR = 'src/maps/' 

"""
Given a wanted SNR and the true signal, return the std that need to have the noise.
"""
def find_sigma_noise(snr, x_ref):
  nref = np.std(x_ref)
  sigma_noise = (10**(-snr/20)) * nref
  return sigma_noise

def find_sigma_noise_cheng(snr, x):
    variance =  ( np.std(x)**2) / (10 ** (snr/10) )
    return np.sqrt(variance)

def create_or_load_knn_rotation_invariant(name, N, image_res, images, K, snr=None, save=True):
    aspireknn_filename = ''
    if snr is None:
        aspireknn_filename = DATA_DIR + 'aspire_knn_{0}_{1}_{2}_{3}.npz'.format(name, N, image_res, K)
    else:
        aspireknn_filename = DATA_DIR + 'aspire_knn_{0}_{1}_{2}_{3}_{4}.npz'.format(name, N, image_res, K, snr)

    classes = None
    reflections = None
    rotations = None
    shifts = None
    correlations = None

    if path.isfile(aspireknn_filename):
        loaded = np.load(aspireknn_filename)
        classes = loaded['classes']
        reflections = loaded['reflections']
        rotations = loaded['rotations']
        correlations = loaded['correlations']
    else:
        print("cannot find {0}, aspire_knn will be executed.".format(aspireknn_filename))
        classes, reflections, rotations, shifts, correlations = aspire_knn_with_rotation_invariant_distance(images, K)
        if save:
            np.savez_compressed(aspireknn_filename, classes=classes, reflections=reflections, rotations=rotations, correlations=correlations)

    return classes, reflections, rotations, correlations


def voxelSaveAsMap(voxel, location = 'out.map'):
    grid = gemmi.FloatGrid(voxel.shape[0],voxel.shape[1],voxel.shape[2] )
    #grid
    arr = np.array(grid, copy=False)
    for i in range(voxel.shape[0]):
        for j in range(voxel.shape[1]):
            for k in range(voxel.shape[2]):
                arr[i,j,k] = voxel[i,j,k]
    
    #for i in range(voxel.shape[0])
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = gemmi.FloatGrid(arr)
    ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name('P1')
    ccp4.grid.unit_cell.set(20,20,20,90,90,90)
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(location)


"""
Normalize midrange dimension vice.
input x:  (N, ndim)

"""
def normalize_midrange(x, ndim=3):
    assert x.shape[1] == ndim , "dimension must match"

    out = np.zeros(x.shape)

    for i in range(ndim):
        out[:,i] = x[:,i] - (x[:,i].max() + x[:,i].min()) / 2

def getAngle(P,Q):
    R = np.dot(P,Q.T)
    theta = (np.trace(R) -1)/2
    return np.arccos(theta) * (180/np.pi)

def reconstruct_result_cheng(V, n, img_size, snr, k):
     # create aspire volume and downsample to image size
    aspire_vol = Volume(V)
    #aspire_vol = aspire_vol.downsample(img_size)

    # return values:
    rotation_angles =  random_rotation_3d(n)

    # determine noise
    noise_variance = find_sigma_noise(snr, aspire_vol.__getitem__(0))

    print(f"noise sigma: {noise_variance}")
    noise_variance = 1e-5

    # create aspire simulation
    sim = create_simulation(aspire_vol, n, rotation_angles, noise_variance)

    # get clean graph
    distance, classes, angles,reflection = rotation_invariant_knn(sim.clean_images(0, n).asnumpy(), K=k)
    clean_graph = Knn(distance, classes, angles, reflection)
    #clean_graph = None

    # get noisy graph
    #distance, classes, angles,reflection = rotation_invariant_knn(sim.images(0, n).asnumpy(), K=k)
    noisy_graph = None #Knn(distance, classes, angles, reflection)

    return aspire_vol, sim, clean_graph, noisy_graph

def create_or_load_dataset_from_map(expertiment_name, map_file, n, img_size, snr, k, normalize=False, ctf=None):
    # checks
    assert path.isfile(MAP_DIR + map_file) , "could not find map file" 
    
    # folders and files
    
    map_file_path = MAP_DIR + map_file
    expertiment_folder = DATA_DIR + expertiment_name + "/"
    rotation_file = expertiment_folder + f"rotations_{n}.npz"
    graph_file_path = expertiment_folder + f"knn_{n}_{img_size}_{k}_{snr}.npz"
    noisy_file_path = expertiment_folder + f"knn_noisy_{n}_{img_size}_{k}_{snr}.npz"
    #'aspire_knn_{0}_{1}_{2}_{3}.npz'.format(name, N, image_res, K)

    if not path.isdir(DATA_DIR + expertiment_name):
        mkdir(expertiment_folder)

    # load map file
    v_npy = mrcfile.open(map_file_path ).data.astype(np.float64)

    # normalize
    if normalize:
        v_npy = normalize_min_max(v_npy)

    # create aspire volume and downsample to image size
    aspire_vol = Volume(v_npy)
    aspire_vol = aspire_vol.downsample(img_size)

    # return values:
    rotation_angles = create_or_load_rotations(rotation_file, n)

    # determine noise
    noise_variance = find_sigma_noise(snr, aspire_vol)
    #found noise_variance: 0.12354835437561838
    print(f"found noise_variance: {noise_variance}")

    noise_variance = find_sigma_noise_cheng(snr, aspire_vol)
    #found noise_variance: 0.12354835437561838
    print(f"found noise_variance: {noise_variance}")

    # noise_variance = 1e-6

    # create aspire simulation
    sim = create_simulation(aspire_vol, n, rotation_angles, noise_variance, ctf=ctf)

    # sim.projections(0,4).show()

    # get clean graph
    clean_graph = create_or_load_knn(graph_file_path, sim.clean_images(0, n).asnumpy(), k)

    # get noisy graph
    noisy_graph = create_or_load_knn(noisy_file_path, sim.images(0, n).asnumpy(), k)

    return aspire_vol, sim, clean_graph, noisy_graph


def create_or_load_knn(filepath, images, k):
    distance = None
    classes = None
    angles = None
    reflection = None

    if path.isfile(filepath):
        loaded = np.load(filepath)
        distance = loaded['distance']
        classes = loaded['classes']
        angles = loaded['angles']
        reflection = loaded['reflection']
        print(f"graph loaded!{filepath}")
    else:
        distance, classes, angles, reflection = rotation_invariant_knn(images, K=k)
        np.savez_compressed(filepath, distance=distance, classes=classes, angles=angles, reflection=reflection)
    
    return Knn(distance, classes, angles, reflection)

    
def create_or_load_rotations(rotation_file, n):
    # handle rotation angles
    rotation_angles = None
    if path.isfile(rotation_file):
        loaded = np.load(rotation_file)
        rotation_angles = loaded['rotation_angles']
        print("rotations loaded!")
    
    if rotation_angles is None:
        rotation_angles = random_rotation_3d(n)
        np.savez_compressed(rotation_file, rotation_angles=rotation_angles)

    return rotation_angles

class Knn:
    def __init__(self, distance, classes, angles, reflection):
        self.distance = distance
        self.classes = classes
        self.angles = angles
        self.reflection = reflection
