from scipy.spatial.transform import Rotation as R
from scipy.ndimage.interpolation import rotate
import numpy as np
from tqdm import tqdm

def my_euler_sequence():
    return 'ZYZ'

def get_2d(d3_points, resolution=100, xlim=[-2,2], ylim=[-2,2]):
    [xlim_l,xlim_u]=xlim
    [ylim_l,ylim_u]=ylim
    devx=(xlim_u-xlim_l)/resolution
    devy=(ylim_u-ylim_l)/resolution
    data_2d=d3_points[:,:2].T
    data_2d.shape[0]
    img=np.zeros((resolution,resolution))
    for i in range(data_2d.shape[1]):
        xi=int((data_2d[0,i]-xlim_l)/devx)
        yi=int((data_2d[1,i]-ylim_l)/devy)
        img[xi,yi]=img[xi,yi]+1
    return img

def rotation3d(d3_points, phi, psi, theta, deg=True):
    r = R.from_euler(my_euler_sequence(), [phi, psi, theta], degrees=deg)
    return r.apply(d3_points)

def random_rotation_3d(N, deg=False):
    return R.random(N).as_euler(my_euler_sequence(), degrees=deg)

def calc_rotation_from_points_on_sphere_2(points):
    n = points.shape[0]
    rotation_point = np.array([1,0,0])
    euler_angles = np.zeros_like(points)
    alpha_beta = np.zeros_like(points)
    
    for i in range(n):
        point=points[i]
        r = np.sqrt(point[0] ** 2 + point[1] ** 2)
        theta = np.sign(point[1]) * np.arccos(point[0] / r)

        
        alpha = np.arccos(point[0] / r) - theta

        intermediate_x = point[0] * np.cos(alpha) - point[1] * np.sin(alpha) 
        intermediate_y = point[0] * np.sin(alpha) + point[1] * np.cos(alpha)
        intermediate_z = point[2]

        r2 =  np.sqrt(intermediate_y ** 2 + intermediate_z ** 2)
        psi = np.sign(intermediate_z) * np.arccos(intermediate_y / r2)
        tau = np.sign(rotation_point[2]) * np.arccos(rotation_point[1] / r2)

        beta = tau - psi

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

        rot = R.__mul__(R.from_matrix(R_x), R.from_matrix(R_z))
        euler_angles[i] = rot.as_euler(my_euler_sequence())
        alpha_beta[i] = np.array([alpha, beta, 0])

    return euler_angles, alpha_beta


def calc_rotation_from_points_on_sphere(points):
    # points: (N, 3)
    n = points.shape[0]
    origin = np.array([1,0,0])
    r = np.sqrt(origin[0] ** 2 + origin[1] ** 2)
    theta = np.sign(origin[1]) * np.arccos(origin[0] / r)

    euler_angles = np.zeros_like(points)
    alpha_beta = np.zeros_like(points)

    for i in range(n):
        point=points[i]
        alpha = np.arccos(point[0] / r) - theta

        intermediate_x = origin[0] * np.cos(alpha) - origin[1] * np.sin(alpha) 
        intermediate_y = origin[0] * np.sin(alpha) + origin[1] * np.cos(alpha)
        intermediate_z = origin[2]

        r2 =  np.sqrt(intermediate_y ** 2 + intermediate_z ** 2)
        psi = np.sign(intermediate_z) * np.arccos(intermediate_y / r2)
        tau = np.sign(point[2]) * np.arccos(point[1] / r2)

        beta = tau - psi

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

        rot = R.__mul__(R.from_matrix(R_x), R.from_matrix(R_z))
        euler_angles[i] = rot.as_euler(my_euler_sequence())
        alpha_beta[i] = np.array([alpha, beta, 0])

    return euler_angles, alpha_beta

def rotate_volume(V, alpha, beta, gamma):
    # we need z y z!
    # angle in degrees!
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html

    alpha = np.rad2deg(alpha)
    beta = np.rad2deg(beta)
    gamma = np.rad2deg(gamma)

    # (2,1) -> x-axis rotation
    # (2,0) -> y-axis rotation
    # (1,0) -> z-axis rotation

    # V = rotate(V, alpha, mode='constant', cval=0, order=3, axes=(1, 0), reshape=False)
    # V = rotate(V, beta, mode='constant', cval=0, order=3, axes=(2, 1), reshape=False)

    V = rotate(V, alpha, mode='constant', cval=0, order=3, axes=(1,0), reshape=False)
    V = rotate(V, beta, mode='constant', cval=0, order=3, axes=(2, 0), reshape=False)
    V = rotate(V, gamma, mode='constant', cval=0, order=3, axes=(1,0), reshape=False)
    return V

def rotate_volume_zxinv(V, alpha, beta, gamma):

    alpha = np.rad2deg(alpha)
    beta = np.rad2deg(beta)
    gamma = np.rad2deg(gamma)

    # (2,1) -> x-axis rotation
    # (2,0) -> y-axis rotation
    # (1,0) -> z-axis rotation

    # V = rotate(V, alpha, mode='constant', cval=0, order=3, axes=(1, 0), reshape=False)
    # V = rotate(V, beta, mode='constant', cval=0, order=3, axes=(2, 1), reshape=False)

    V = rotate(V, alpha, mode='constant', cval=0, order=3, axes=(1,0), reshape=False)
    V = rotate(V, beta, mode='constant', cval=0, order=3, axes=(2, 1), reshape=False)
    V = rotate(V, gamma, mode='constant', cval=0, order=3, axes=(1,0), reshape=False)
    return V

def create_mask3d(resolution):
    x3,y3,z3 = np.indices((resolution,resolution,resolution))
    x3,y3,z3=x3-resolution/2,y3-resolution/2,z3-resolution/2
    mask3d=(x3**2+y3**2+z3**2)<(resolution/2)**2

    return mask3d

def create_r(resolution):
    x2,y2=np.indices((resolution,resolution))
    x2,y2=x2-resolution/2,y2-resolution/2
    r=((resolution/2)**2-x2**2-y2**2)
    r[r<1]=1
    r=np.sqrt(r)
    r=r/r.max()*2

    return r

def reconstruction_naive(estimated_images, n, resolution, rotation_list_re):
    mask3d = create_mask3d(resolution)
    r = create_r(resolution)
    voxel_shape = (resolution,resolution,resolution)

    V = np.zeros(voxel_shape, dtype=np.float)

    for i in tqdm(range(n)):
        estimation = estimated_images[i]
        #V0=np.broadcast_to(image*r,(resolution,resolution,resolution)).transpose((1,2,0))
        V0 = np.broadcast_to(estimation / r ,voxel_shape).transpose((1,2,0))
        V0 = np.array(V0)
        V0[mask3d == False] = 0
        
        rotations = rotation_list_re[i]
        V = V + rotate_volume(V0,rotations[0],rotations[1],rotations[2])
        
    V = V / n

    return V
