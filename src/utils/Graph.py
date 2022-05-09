import numpy as np

def generate_knn_from_distances_with_edges(distanceMatrix, K, sym=True, ordering='desc', dataType='float64', ignoreFirst=False):
    samples = distanceMatrix.shape[0]
    A = np.zeros_like(distanceMatrix)
    neighbors = np.zeros((samples, K))
    edges = set()

    for i in range(samples):
        distanceRow = distanceMatrix[i, :]

        if ordering == 'desc':
          if ignoreFirst:
            index = np.argsort(-distanceRow)[1:K+1]
          else:
            index = np.argsort(-distanceRow)[0:K]
        else:
          if ignoreFirst:
            index = np.argsort(distanceRow)[1:K+1]
          else:
            index = np.argsort(distanceRow)[0:K]

        A [i][index] = 1
        neighbors[i] = index

        for j in index:
          edges.add((i,j))
          edges.add((j,i))

        if sym:
            A  = (((A  + A .T) > 0) * 1 ).astype(dataType)
            

    return A , neighbors.astype(np.int), np.array(list(edges))