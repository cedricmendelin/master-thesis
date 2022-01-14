def read_map(mapname, fid=None):
    """Reads CCP4 type map (.map) or MRC type map.

    Arguments:
        Inputs:
            mapname: string
                CCP4/MRC map file name
        Outputs:
            unit_cell: float, 1D array
                Unit cell
            arr: float, 3D array
                Map values as Numpy array
            origin: list
                Map origin list
     """
    import mrcfile
    import numpy as np

    try:
        file = mrcfile.open(mapname)
        order = (3-file.header.maps, 3-file.header.mapr, 3-file.header.mapc)
        #print('axes order: ', order)
        axes_order = "".join(["ZYX"[i] for i in order])
        print('axes order2: ', axes_order)
        arr = np.asarray(file.data, dtype="float")
        arr = np.moveaxis(a=arr, source=(0,1,2), destination=order)
        if fid is not None:
            fid.write('Axes order: %s\n' % (axes_order))
        unit_cell = np.zeros(6, dtype='float')
        cell = file.header.cella[['x', 'y', 'z']]
        unit_cell[:3] = cell.view(('f4', 3))
        # swapping a and c to compatible with ZYX convension
        unit_cell[0], unit_cell[2] = unit_cell[2], unit_cell[0]
        unit_cell[3:] = float(90)
        origin = [
                1 * file.header.nxstart,
                1 * file.header.nystart,
                1 * file.header.nzstart,
            ]
        file.close()
        print(mapname, arr.shape, unit_cell[:3])
        return unit_cell, arr, origin
    except FileNotFoundError as e:
        print(e)

import numpy as np
from vedo import *
import matplotlib.pyplot as plt
import utils.Plotting as plot

unit_cell, arr, origin = read_map('C:\master-thesis\src\emd_25792.map')

print(unit_cell)
print(arr.shape)
print(origin)

#np.savetxt('C:\master-thesis\emd_test.csv', arr.reshape((8000,8000)))

plot.plot_imshow(arr[200])
plot.plot_imshow(arr[1])
plot.plot_imshow(arr[50])
plot.plot_imshow(arr[300])

#pointcloud = Points(arr[])

input("Enter to terminate")