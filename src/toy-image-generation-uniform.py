import matplotlib.pyplot as plt

rect_min = 60
rect_max = 260
rect_w_h = 80

circle_min = 100
circle_max = 300
circle_r = 40

triangle_x1_min = 60
triangle_x1_max = 260



my_dpi = 100

samples = 10000

plt.style.use('dark_background')

import numpy as np
rng = np.random.default_rng(12345)



for i in range(samples):

    plt.figure(figsize=(5.2, 5.2), dpi=my_dpi)
    #plt.axes()
    circle_vals = rng.integers(low=circle_min, high=circle_max, size=2)
    circle = plt.Circle((circle_vals[0], circle_vals[1]), radius=circle_r, fc='white')
    plt.gca().add_patch(circle)

    rectangle_vals = rng.integers(low=rect_min, high=rect_max, size=2)
    rectangle = plt.Rectangle((rectangle_vals[0], rectangle_vals[1]), rect_w_h, rect_w_h, fc='white')
    plt.gca().add_patch(rectangle)

    triangle_x1 = rng.integers(low=triangle_x1_min, high=triangle_x1_max, size=2)
    points = [[triangle_x1[0], triangle_x1[1]], [triangle_x1[0] + 40, triangle_x1[1] + 80], [triangle_x1[0] + 80, triangle_x1[1]]]
    polygon = plt.Polygon(points, fc='white')
    plt.gca().add_patch(polygon)

    plt.axis('scaled')
    plt.axis('off')
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.plot()
    plt.savefig(f'src/toyimages_uniform/image_{i}', dpi=my_dpi, bbox_inches='tight', pad_inches=0)
    #plt.show()
    plt.close()
