from utils.Plotting import * 
import numpy as np
from skimage.draw import polygon, rectangle, disk
import time

def uniform_unit_circle_sample(samples):
    length = np.sqrt(np.random.uniform(0, 1, samples))
    angle = np.pi * np.random.uniform(0, 2, samples) 

    x = length * np.cos(angle)
    y = length * np.sin(angle)

    return x, y

def scaled_circle_samples(distribution_circle_diamter, shift, samples):
    xs, ys = uniform_unit_circle_sample(samples)  #* distribution_circle_radius
    xs, ys = (xs + 1) / 2, (ys + 1 ) / 2
    xs, ys = xs * distribution_circle_diamter + shift, ys * distribution_circle_diamter + shift
    xs, ys = np.ceil(xs), np.ceil(ys)

    return xs, ys

def draw_uniform_toyimages(resolution, shape_size, samples):
    circle_center_boundary = resolution - (shape_size)
    rect_outer_circle_diamter = np.sqrt(shape_size ** 2 + shape_size ** 2)
    rect_center_boundary = (resolution - rect_outer_circle_diamter)
    triangle_center_boundary = rect_center_boundary

    half_shape_size = (shape_size / 2)
    shift_rect = np.ceil(rect_outer_circle_diamter ) / 2
    shift_triangle = shift_rect

    # some numerical drawing issues, add some padding:
    circle_center_boundary = circle_center_boundary - 2
    rect_center_boundary = rect_center_boundary - 6
    shift_rect = shift_rect + 3

    circle_x, circle_y = scaled_circle_samples(circle_center_boundary, half_shape_size, samples)
    
    rect_x, rect_y = scaled_circle_samples(rect_center_boundary, shift_rect, samples)
    rect_start = np.array([(rect_x - half_shape_size ).astype(np.int) , (rect_y - half_shape_size).astype(np.int)]).T 
    rect_end = np.array([(rect_x + half_shape_size).astype(np.int), (rect_y + half_shape_size).astype(np.int)]).T 
    print(rect_start.shape)

    traingle_x, triangle_y = scaled_circle_samples(triangle_center_boundary, shift_triangle, samples)
    triangle_x1 = (traingle_x - half_shape_size)
    triangle_x3 = (traingle_x + half_shape_size)

    triangle_y1_y3 = (triangle_y - half_shape_size)
    triangle_y2 = (triangle_y + half_shape_size)
        
    images = np.zeros((samples, resolution, resolution))
    for i in range(samples):
        # circle
        rr, cc =  disk((circle_x[i], circle_y[i]), half_shape_size)
        images[i, rr, cc] = 1

        rr, cc =  rectangle(start= (rect_start[i,0], rect_start[i,1]) , end=(rect_end[i,0], rect_end[i,1]))
        images[i, rr, cc] = 1

        rr, cc = polygon(np.array([triangle_x1[i], triangle_x3[i], traingle_x[i]]), np.array([triangle_y1_y3[i], triangle_y1_y3[i], triangle_y2[i]]))
        images[i, rr, cc] = 1

    return images
    #return np.clip(images, 0, 1)

t = time.time()
# res = 64
# shape_size = 16


images = draw_uniform_toyimages(64, 16, 1000)
from matplotlib import patches
print (" time", time.time() - t)

for i in range(10):
    fig, ax = plt.subplots(1)
    ax.imshow(images[i])
    ax.axis('off')
    circ = patches.Circle((32, 32), 32, color='red', linewidth=1, fill=False)
    ax.add_patch(circ)

plt.show()

