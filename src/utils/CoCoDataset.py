import numpy as np
import random
import imageio
import os

## Returns grayscale images at random in a given path and list of files output of size number x 1 x N1 x N2
def load_images_files(path,files, N1, N2, number=1, num_seed = None):
    random.seed(a=num_seed) # if None, system clock
    result = np.zeros((number, N1, N2))
    
    for i in range(number):
        M1 = 0
        M2 = 0
        while M1<N1 or M2<N2:
            d=random.choice(files)
            im = imageio.imread(path+d)
            if im.ndim > 2: # passes to grayscale if color
                im = im[:,:,0]
            [M1,M2] = im.shape
        im = im[0:N1,0:N2]/255
        result[i, :, :] = im
    return result


# # usage:
# image_path = "src/data/val2017/"
# files = os.listdir(image_path)
# x = load_images_files(image_path, files, N1, N2, number=batch_size)