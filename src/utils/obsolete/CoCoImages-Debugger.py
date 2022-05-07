from PIL import Image
import numpy as np
import random
import imageio
import os

from skimage.transform import rescale
from matplotlib import cm

def load_images_files(path, files, N1, N2, number=1, circle_padding=False, num_seed=None):
  # result = np.zeros((number, N1, N2))

  for i in range(number):
      file =files[i]
      im = imageio.imread(path + file)
      if im.ndim > 2:  # passes to grayscale if color
          im = im[:, :, 0]

      [M1, M2] = im.shape
      im = rescale(im, scale=(N1 / M1, N2 / M2),mode='reflect', multichannel=False)

      # while M1 < N1 or M2 < N2:
      #     file =files[i]
      #     im = imageio.imread(path + file)
      #     #files.remove(file)
      #     if im.ndim > 2:  # passes to grayscale if color
      #         im = im[:, :, 0]
      #     [M1, M2] = im.shape
      # im = im[0:N1, 0:N2]/255

      # plt.imsave("src/data/val2017-grey-withoutpad/" + file, im, cmap = cm.gray)

      if circle_padding:
          _res: int = int(np.sqrt( N1 ** 2 / 2))
          scaleX = _res / N1
          scaleY = _res / N2
          im = rescale(im, scale=(scaleX, scaleY),
                        mode='reflect', multichannel=False)

          padding = int((N1 - _res) / 2)
          p = (padding, padding)
          if padding + padding + _res - N1 == -1:
            p = (padding + 1, padding)

          im = np.pad(im, [p, p], mode='constant', constant_values=0)
      plt.imsave("src/data/val2017-grey2/" + file, im, cmap = cm.gray)

      # result[i, :, :] = im
  return

import matplotlib.pyplot as plt

image_path = "src/data/val2017/"
files = os.listdir(image_path)
RESOLUTION=200

load_images_files(image_path, files, RESOLUTION, RESOLUTION, number=10, circle_padding=True)


# for i in range(5000):
#   plt.imsave("src/data/val2017-grey/" + files[i], x[i], cmap = cm.gray)
  # im = Image.fromarray(x[i]).convert('RGB')
  # im.save("src/data/val2017-grey/" + files[i])


