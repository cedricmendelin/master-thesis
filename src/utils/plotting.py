import matplotlib.pyplot as plt
color_map = plt.cm.get_cmap('gray')
reversed_color_map = color_map.reversed()
plt.ion()

def plot_imshow(data):
    plt.ion()
    plt.figure()
    plt.imshow(data, cmap=reversed_color_map)



def plot_3dscatter(x,y,z, figsize):
    #Plot with color to visualize the manifold.
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(x, y, z, cmap='hsv')
    plt.show()
