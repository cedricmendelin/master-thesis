from vedo import Mesh,dataurl
from tqdm import tqdm
from skimage.transform import rotate
from scipy.spatial.transform import Rotation as R
import sys
sys.path.insert(0, '..')

#magnolia.vtk
#panther.stl
#bunny.obj
mesh = Mesh(dataurl+"bunny.obj").normalize().subdivide()

mesh.show()