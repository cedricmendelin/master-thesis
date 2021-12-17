import matplotlib.pyplot as plt

color_map = plt.cm.get_cmap('gray')
reversed_color_map = color_map.reversed()

def plot_imshow(data):
    plt.figure()
    plt.imshow(data, cmap=reversed_color_map)
