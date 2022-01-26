import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use('TkAgg')
if(not plt.isinteractive):
    plt.ion()

color_map = plt.cm.get_cmap('gray')
reversed_color_map = color_map.reversed()

def plot_imshow(data):
    plt.figure()
    plt.imshow(data, cmap=reversed_color_map)

def plot_3dscatter(x,y,z, figsize):
    #Plot with color to visualize the manifold.
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(x, y, z, cmap='hsv')
    plt.show()

def draw_graph(A):
    g = nx.from_numpy_matrix(A)
    nx.draw(g)

def plot_voxels(voxels):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels)
    ax.view_init(30, 150)
    plt.show()