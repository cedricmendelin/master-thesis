import numpy as np
from utils.Plotting import *


def sampling_sphere(Ntheta):
    th = np.random.random(Ntheta) * np.pi * 2 # random inverval [0, 2pi]
    x = np.random.random(Ntheta) * 2 - 1 # random interval [1, -1]
    out = np.array([np.cos(th) * np.sqrt(1 - x**2), np.sin(th) * np.sqrt(1 - x**2),x]).T
    return out


def fibonacci_sphere(samples=1000):
    points = np.zeros((samples, 3))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        x = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        
        theta = phi * i  # golden angle increment [0, 2pi]
        r = np.sqrt(1 - x**2)
        points[i] = np.array([np.cos(theta) * r, np.sin(theta) * r,x])
    return points


sphere = sampling_sphere(1000)
plot_3d_scatter(sphere)


fibo_sphere = fibonacci_sphere(1000)
plot_3d_scatter(fibo_sphere)


theta1 = np.arctan2(fibo_sphere[:,1], fibo_sphere[:,2])
theta2 = np.arctan(fibo_sphere[:,0], fibo_sphere[:,2])

print(theta1)
print(theta2)

input("enter to terminate")

