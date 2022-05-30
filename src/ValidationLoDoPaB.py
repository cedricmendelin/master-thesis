import numpy as np
import matplotlib.pyplot as plt

from utils.Plotting import plot_imshow


test_data = "src/data/limited-CT/data_test.npz"

np_file = np.load(test_data)

for data in np_file:
  print(data)


labels = np_file['label']
proj = np_file['proj']

plot_imshow(labels[0])
plot_imshow(proj[0])


print(proj.shape)
plt.show()
