import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use('TkAgg')
if(not plt.isinteractive):
    plt.ion()

color_map = plt.cm.get_cmap('gray')
reversed_color_map = color_map.reversed()

def plot_imshow(data, title='', show=False, colorbar=True, aspect=None, c_map = color_map):
    plt.figure()
    plt.imshow(data, cmap=c_map, aspect=aspect)
    plt.title(title)
    if colorbar:
        plt.colorbar()
    if show:
        plt.show()

def plot_image_grid(images, titles, col=2, row= 1, colorbar=True, aspect=None, c_map = color_map):
    fig = plt.figure(figsize=(8, 8))
    for i in range(0, col*row):
        fig.add_subplot(row, col, i+1)
        plt.imshow(images[i], cmap=c_map, aspect=aspect)
        plt.title(titles[i])

def plot_2d_scatter(data, figsize=(10,10), title='', show=False):
    plot_2dscatter(data[:,0], data[:,1], figsize, title, show)

def plot_2dscatter(x, y, figsize=(10,10), title='', show=False):
    fig = plt.figure(figsize=figsize)
    
    # Creating plot
    plt.scatter(x, y, cmap='hsv')
    plt.title(title)

    if show:
        plt.show()

def plot_3d_scatter(data, figsize=(10,10), title='', show=False):
    plot_3dscatter(data[:,0], data[:,1], data[:,2], figsize, title, show)

def plot_3dscatter(x,y,z, figsize=(10,10), title='', show=False):
    #Plot with color to visualize the manifold.
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(x, y, z, cmap='hsv')
    ax.set_title(title)

    if show:
        plt.show()

def draw_graph(A):
    g = nx.from_numpy_matrix(A)
    nx.draw(g)

def plot_voxels(voxels):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels)
    ax.view_init(30, 150)
    plt.show()