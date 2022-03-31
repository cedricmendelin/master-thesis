from sklearn.metrics.pairwise import haversine_distances
import numpy as np
import time
from utils.Plotting import *
def great_circle_distance(lon1, lat1, lon2, lat2, debug_i):
    # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    return np.arccos( max(min(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2), 1), -1))


N = 6
angles = np.linspace(0,2 * np.pi, N)

points = np.array([np.cos(angles), np.sin(angles)]).T
plot_2d_scatter(points)
t = time.time()
dis1 = np.array([great_circle_distance(points[i,0], points[i,1], points[j,0], points[j,1], i) for i in range(N) for j in range(N)]).reshape((N,N))
print("dis1-:", t-time.time())
t = time.time()
dis2 = haversine_distances(points, points)
print("dis2-:",t-time.time())

print(dis1.shape)
print(dis2.shape)
print(dis1)
print(dis2)

print(np.linalg.norm(dis1-dis2))


plt.show()