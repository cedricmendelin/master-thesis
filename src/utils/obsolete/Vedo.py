from vedo import Mesh, dataurl, mesh2Volume
from .obsolete.Data import *
from aspire.volume import Volume
import vedo

DATA_DIR = 'src/data/'

"""
Create bunny dataset.
"""
def create_or_load_bunny_immages(N, image_size, snr, save=True):
    return create_or_load_vedo_images(dataset=dataurl+"bunny.obj", name='bunny', N=N, image_size=image_size, snr=snr, save=save)



"""
gets the vedo dataset signal, which is by default midrange normalized
"""
def get_vedo_dataset_signal(dataset):
    mesh = Mesh(dataset).normalize().subdivide()

    #pts = mesh.points(copy=True)  # pts is a copy of the points not a reference
    pts = mesh.points()
    #print(pts.max()) # 2.8440073
    #print(pts.min()) # -1.4367026

    # why midrange?

    pts=pts[np.random.permutation(pts.shape[0]),:]

    pts[:,0]=pts[:,0]-(pts[:,0].max()+pts[:,0].min())/2
    pts[:,1]=pts[:,1]-(pts[:,1].max()+pts[:,1].min())/2
    pts[:,2]=pts[:,2]-(pts[:,2].max()+pts[:,2].min())/2
    pts_max=np.linalg.norm(pts,axis=1).max()
    pts=pts/pts_max

    return pts


"""
Create vedo dataset by hand.
"""
def create_or_load_vedo_images(dataset, name, N, image_size, snr, save=True):
    image_filename = DATA_DIR + 'original_images_{0}_{1}_{2}.npz'.format(name, N, image_size)
    noisy_filename = DATA_DIR + 'noisy_images_{0}_{1}_{2}_{3}.npz'.format(name, N, image_size, snr)

    # original images rotated
    original_images = None
    rotations = None
    # load original images if already created
    if path.isfile(image_filename):
        loaded = np.load(image_filename)
        original_images = loaded['original_images']
        rotations = loaded['rotations']

    else:
        print("start generating dataset")
        signal = get_vedo_dataset_signal(dataset)

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
        print("start generating dataset noisy")
        sigma = find_sigma_noise(snr, get_vedo_dataset_signal(dataset))
        mu = 0
        noise = np.random.normal(mu, sigma, (N, image_size, image_size))
        
        noisy_images = original_images + noise

        if save:
            np.savez_compressed(noisy_filename, noisy_images=noisy_images, noise=noise, snr=snr)

    return original_images, rotations, noisy_images, noise


def visualize_voxels_3d(voxels):
    vol = vedo.Volume(voxels)
    vedo.show(vol)

def vedo_bunny_to_asipre_volume():
    mesh = Mesh(dataurl + "bunny.obj")#.normalize().subdivide()
    vol = mesh2Volume(mesh, spacing=(0.01,0.01,0.01))

    vol_equally_spaced = np.pad(vol.tonumpy(), ((0,3),(0,6),(0,57)), mode='constant', constant_values=0).astype(np.float32)

    return Volume(vol_equally_spaced)

def aspire_volume_to_vedo_volume(aspire_vol, vol_index=0):
    return vedo.Volume(aspire_vol.__getitem__(vol_index))