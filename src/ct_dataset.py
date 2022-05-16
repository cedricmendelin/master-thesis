import numpy as np

from utils.Plotting import *
from utils.SNRHelper import *

data_folder = "data/limited-CT/"
horizontal_cts_path = data_folder +  "horizontal_snr25.0.npz"
vertical_cts_path = data_folder + "vertical_snr25.0.npz"

horizontal_cts = np.load(horizontal_cts_path)
for k in horizontal_cts.files:
  print (k)
print(horizontal_cts)

x_train = horizontal_cts["x_train"]
x_test = horizontal_cts["x_test"]
y_train = horizontal_cts["y_train"]
print(x_train.shape)
print(x_test.shape)
print(x_train[150,:,:].shape)

print(x_train[0:1024].shape)

plot_imshow(x_train[150])
plot_imshow(y_train[150])
plot_imshow(add_noise_np(25, x_train[150].reshape((64,64))))
plot_imshow(add_noise_np(5, x_train[150].reshape((64,64))))

vertical_cts = np.load(vertical_cts_path)
x_train2 = vertical_cts["x_train"]
y_train2 = vertical_cts["y_train"]
print(x_train2.shape)
print(y_train2.shape)

plot_imshow(x_train2[150])
plot_imshow(y_train2[150])


plt.show()