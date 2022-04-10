import numpy as np

from utils.obsolete.Data import sampling_sphere, set_small_values_to
from utils.Geometry import calc_rotation_from_points_on_sphere_ZYZ

from scipy.spatial.transform import Rotation as R




n = 1000
normal_dist_sphere = sampling_sphere(n)
rotation_point=np.array([0,1/np.sqrt(2),1/np.sqrt(2)])

rotations = calc_rotation_from_points_on_sphere_ZYZ(normal_dist_sphere, n, debug=True, rotation_point=rotation_point)
print(rotations)

print(rotations.shape)

for i in range(n):
  if i % 100 == 0:
    p = R.from_euler("ZYZ", rotations[i]).apply(normal_dist_sphere[i])
    p = set_small_values_to(p)
    print(f"Actual {rotation_point}, calculated {p}, diff{rotation_point - p}")
