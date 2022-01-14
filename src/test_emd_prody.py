from prody import *
import numpy as np
from vedo import Mesh

#emd = parseEMD('25792', cutoff=1.2, n_nodes=8000)
#writePDB('25792.pdb', emd)



pdb = parsePDB('25792.pdb')
cords = pdb.getCoords()

print(cords.shape)


