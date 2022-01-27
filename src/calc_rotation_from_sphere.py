import numpy as np

from utils.Data import *

from scipy.spatial.transform import Rotation as R


p1 = np.array([1,0,0])
p2 = np.array([0,-1/np.sqrt(2),1/np.sqrt(2)])

# rotation z-axis
r = np.sqrt(p1[0] ** 2 + p1[1] ** 2)
theta = np.sign(p1[1]) * np.arccos(p1[0] / r)

print(r)
print(theta)
print(f"x should be {r * np.cos(theta)}")
print(f"y should be {r * np.sin(theta)}")

alpha = np.arccos(p2[0] / r) - theta
alpha_degree = alpha * (180 / np.pi)

print("alpha for z rotation:")
print(alpha)
print(alpha * (180 / np.pi))

print(f"d should be {r * np.cos(alpha + theta)}")
print(f"d should be {p1[0] * np.cos(alpha) - p1[1] * np.sin(alpha)}")

intermediate_x = p1[0] * np.cos(alpha) - p1[1] * np.sin(alpha) 
intermediate_y = p1[0] * np.sin(alpha) + p1[1] * np.cos(alpha)
intermediate_z = p1[2]

print(f"intermediate point{intermediate_x}/{intermediate_y}/{intermediate_z}")

print(f"g should be {intermediate_y}")
print(f"h should be {intermediate_z}")

# rotation x-axis

r2 =  np.sqrt(intermediate_y ** 2 + intermediate_z ** 2)

psi = np.sign(intermediate_z) * np.arccos(intermediate_y / r2)
tau = np.sign(p2[2]) * np.arccos(p2[1] / r2)

psi_2 = np.sign(intermediate_y) * np.arcsin(intermediate_z / r2)
tau_2 = np.sign(p2[1]) * np.arcsin(p2[2] / r2)

print(f"g should be {r2 * np.cos(psi)}")
print(f"h should be {r2 * np.sin(psi)}")

print(f"psi should be same {psi} = {psi_2}")
print(f"tau should be same {tau} = {tau_2}")

psi_degree = psi * (180 / np.pi)
tau_degree = tau * (180 / np.pi)

beta = tau - psi

print("beta for x rotation:")
print(beta)
print(beta * (180 / np.pi))

final_x = intermediate_x
final_y = intermediate_y * np.cos(beta) - intermediate_z * np.sin(beta) 
final_z = intermediate_y * np.sin(beta) + intermediate_z * np.cos(beta) 


R_z = np.array(
  [
    [np.cos(alpha), - np.sin(alpha),0],
    [np.sin(alpha), np.cos(alpha),0],
    [0,0,1]
  ])

R_x = np.array(
  [
    [1,0,0], 
    [0, np.cos(beta), - np.sin(beta)], 
    [0, np.sin(beta), np.cos(beta)]
  ])


print("final points:")
print(final_x)
print(final_y)
print(final_z)

print("final points with rotation calc:")
R_final = np.multiply(R_z,  R_x)
p_final = R_x.dot(R_z.dot(p1))
#p_final = R_final.dot(p1) 

print(p_final[0])
print(p_final[1])
print(p_final[2])

R_z = set_small_values_to(R_z)
R_x= set_small_values_to(R_x)

rot = R.__mul__(R.from_matrix(R_x), R.from_matrix(R_z))

final_rot_bla = rot.apply(p1)

print(final_rot_bla)

print("euler angles")
print(rot.as_euler('zxz'))

rot2 = R.from_euler("zxz", rot.as_euler('zxz'))
final_rot_bla = rot2.apply(p1)

set_small_values_to(final_rot_bla)

print(final_rot_bla)


assert np.abs(p2[0]) <= r , "d must be smaller or equal r"