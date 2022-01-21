import numpy as np
from vedo import Mesh, dataurl
from .Geometry import *
from .Graph import *
import os.path as path
import gemmi

basedir = 'src/data/'

"""
Given a wanted SNR and the true signal, return the std that need to have the noise.
"""
def find_sigma_noise(snr, x_ref):
  nref = np.std(x_ref)
  sigma_noise = (10**(-snr/20)) * nref
  return sigma_noise

"""
Create bunny dataset.
"""
def create_or_load_bunny_immages(N, image_size, snr, save=True):
    return create_or_load_vedo_images(dataset=dataurl+"bunny.obj", name='bunny', N=N, image_size=image_size, snr=snr, save=save)


def check_is_file():
    if path.isfile(basedir + 'original_images_bunny_100_100.npz'):
        print('success')
    else:
        print('nope')

def get_dataset_signal(dataset):
    mesh = Mesh(dataset).normalize().subdivide()

    #pts = mesh.points(copy=True)  # pts is a copy of the points not a reference
    pts = mesh.points()
    #print(pts.max()) # 2.8440073
    #print(pts.min()) # -1.4367026

    # why midrange?
    pts[:,0]=pts[:,0]-(pts[:,0].max()+pts[:,0].min())/2
    pts[:,1]=pts[:,1]-(pts[:,1].max()+pts[:,1].min())/2
    pts[:,2]=pts[:,2]-(pts[:,2].max()+pts[:,2].min())/2

    return pts

"""
Create vedo dataset
"""
def create_or_load_vedo_images(dataset, name, N, image_size, snr, save=True):
    image_filename = basedir + 'original_images_{0}_{1}_{2}.npz'.format(name, N, image_size)
    noisy_filename = basedir + 'noisy_images_{0}_{1}_{2}_{3}.npz'.format(name, N, image_size, snr)

    # original images rotated
    original_images = None
    rotations = None
    # load original images if already created
    if path.isfile(image_filename):
        loaded = np.load(image_filename)
        original_images = loaded['original_images']
        rotations = loaded['rotations']

    else:
        signal = get_dataset_signal(dataset)

        rotations = random_rotation_3d(N)
        original_images = np.zeros((N, image_size, image_size))

        for id in range(N):
            original_images[id] = get_2d(rotation3d(signal,rotations[id, 0], rotations[id, 1], rotations[id, 2]), resolution=image_size)
        
        if save:
            np.savez_compressed(image_filename, original_images=original_images, rotations=rotations)
        
    # noisy images:
    noisy_images = None
    noise = None

    if path.isfile(noisy_filename):
        loaded = np.load(noisy_filename)
        noisy_images = loaded['noisy_images']
        noise = loaded['noise']
    else:
        sigma = find_sigma_noise(snr, get_dataset_signal(dataset))
        mu = 0
        noise = np.random.normal(mu, sigma, (N, image_size, image_size))
        
        noisy_images = original_images + noise

        if save:
            np.savez_compressed(noisy_filename, noisy_images=noisy_images, noise=noise, snr=snr)

    return original_images, rotations, noisy_images, noise


def create_or_load_knn_rotation_invariant(name, N, image_res, images, K, snr=None, save=True):
    aspireknn_filename = ''
    if snr is None:
        aspireknn_filename = basedir + 'aspire_knn_{0}_{1}_{2}_{3}.npz'.format(name, N, image_res, K)
    else:
        aspireknn_filename = basedir + 'aspire_knn_{0}_{1}_{2}_{3}_{4}.npz'.format(name, N, image_res, K, snr)

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