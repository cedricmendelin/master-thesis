import numpy as np
import random
import imageio

from skimage.transform import rescale


# Returns grayscale images at random in a given path and list of files output of size number x 1 x N1 x N2
def load_images_files(path, files, N1, N2, number=1, circle_padding=False, num_seed=None):
    random.seed(a=num_seed)  # if None, system clock
    result = np.zeros((number, N1, N2))
    assert N1 == N2, "resolution of images need to be same"

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
            _res: int = int(np.sqrt(N1 ** 2 / 2))
            scaleX = _res / N1
            scaleY = _res / N2
            im = rescale(im, scale=(scaleX, scaleY),
                         mode='reflect', multichannel=False)
            padding = int((N1 - _res) / 2)
            p = (padding, padding)
            im = np.pad(im, [p, p], mode='constant', constant_values=0)

        result[i, :, :] = im
    return result


def load_images_files_rescaled(path, files, N1, N2, number=1, circle_padding=False, num_seed=None):
    random.seed(a=num_seed)  # if None, system clock
    result = np.zeros((number, N1, N2))
    assert N1 == N2, "resolution of images need to be same"

    for i in range(number):
        file = files[i]
        im = imageio.imread(path + file)
        if im.ndim > 2:  # passes to grayscale if color
            im = im[:, :, 0]

        [M1, M2] = im.shape
        im = rescale(im, scale=(N1 / M1, N2 / M2),
                     mode='reflect', multichannel=False)

        if circle_padding:
            _res: int = int(np.sqrt(N1 ** 2 / 2))
            scaleX = _res / N1
            scaleY = _res / N2
            im = rescale(im, scale=(scaleX, scaleY),
                         mode='reflect', multichannel=False)

            padding = int((N1 - _res) / 2)
            p = (padding, padding)
            if padding + padding + _res - N1 == -1:
                p = (padding + 1, padding)

            im = np.pad(im, [p, p], mode='constant', constant_values=0)
        result[i, :, :] = im
    return result
