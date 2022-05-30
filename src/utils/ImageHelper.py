import numpy as np
import random
import imageio

from skimage.transform import rescale


# Returns grayscale images at random in a given path and list of files output of size number x 1 x N1 x N2
def load_images_files(path, files, N1, N2, number=1, circle_padding=False, num_seed=None):
    random.seed(num_seed)  # if None, system clock
    result = np.zeros((number, N1, N2))
    assert N1 == N2, "resolution of images need to be same"
    lin = np.linspace(-1,1, N1)
    XX, YY = np.meshgrid(lin,lin)
    circle = ((XX**2+YY**2)<=1)*1.

    for i in range(number):
        M1 = 0
        M2 = 0
        while M1 < N1 or M2 < N2:
            file = random.choice(files)
            im = imageio.imread(path + file)
            files.remove(file)
            if im.ndim > 2:  # passes to grayscale if color
                im = im[:, :, 0]
            [M1, M2] = im.shape
        im = im[0:N1, 0:N2]/255

        if circle_padding:
           im = im * circle

        result[i, :, :] = im
    return result


def load_images_files_rescaled(path, files, N1, N2, number=1, circle_padding=False, num_seed=None):
    random.seed(a=num_seed)  # if None, system clock
    result = np.zeros((number, N1, N2))
    assert N1 == N2, "resolution of images need to be same"
    lin = np.linspace(-1,1, N1)
    XX, YY = np.meshgrid(lin,lin)
    circle = ((XX**2+YY**2)<=1)*1.

    for i in range(number):
        file = files[i]
        im = imageio.imread(path + file)
        if im.ndim > 2:  # passes to grayscale if color
            im = im[:, :, 0]

        [M1, M2] = im.shape
        im = rescale(im, scale=(N1 / M1, N2 / M2),
                     mode='reflect', multichannel=False)

        if circle_padding:
           im = im * circle
        result[i, :, :] = im
    return result


def add_circle_padding_to_images(images):
    res = images.shape[1]
    lin = np.linspace(-1,1,res)
    XX, YY = np.meshgrid(lin,lin)
    circle = ((XX**2+YY**2)<=1)*1.

    result = np.zeros_like(images)

    for i in range(images.shape[0]):
        result[i] = images[i] * circle

    return result
