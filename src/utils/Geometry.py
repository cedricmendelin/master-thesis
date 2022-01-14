from scipy.spatial.transform import Rotation as R
import numpy as np

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
    r = R.from_euler('xyz', [phi, psi, theta], degrees=deg)
    return r.apply(d3_points)

def random_rotation_3d(N, deg=True):
    return R.random(N).as_euler('xyz', degrees=deg)