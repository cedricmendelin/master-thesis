import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

def generate_knn_from_distances_with_edges(distanceMatrix, K, sym=True, ordering='desc', dataType='float64', ignoreFirst=False):
    """ Generates k-nn graph for given distances.
    Args:
        distanceMatrix (numpy.array): The distances which will be used to determine neighbours. Shape (N, N)
        K int: Defines the number of neighbours per node.
        sym (bool, optional): Defines if the adjecency matrix will by symmetric or not.
        ordering (str, optional): Ordering of distances, can be 'asc' (ascending) or 'desc' (descending). Defaults to 'desc'.
        dataType (str, optional): Datatype for adjenceny matri. Defaults to 'float64'.
        ignoreFirst (bool, optional): If set to True, smallest distance (probably self-loop), will be ignored. Defaults to False.

    Returns:
        A, neighbours, edges: Return a graph in three different representations. 
        The first element is a (N, N) matrix and refers to the adjecency matrix.
        The second element is a set of neighbours (N, K) where for every node, the indices of neighbour nodes are available.
        The third and last element is a list of edges, where edges are tuples (n1, n2)
    """
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

        A[i][index] = 1
        neighbors[i] = index

        for j in index:
            edges.add((i, j))
            edges.add((j, i))

        if sym:
            A = (((A + A .T) > 0) * 1).astype(dataType)

    return A.astype(dataType), neighbors.astype(np.int), np.array(list(edges))

"""
Gets the degree matrix of given matrix.
"""
def get_degree_matrix(M):
  return np.diag(M.sum(axis=1))


"""
Calculate L = D - A and retuns first n smallest eigenvectors ignoring the first one.
"""
def calc_graph_laplacian(A, embedDim=2):
  # Laplacian:
  L = get_degree_matrix(A) - A
  return extract_embedding(L, embedDim)

def extract_embedding(L, embedDim):
  L=sparse.csr_matrix(L)
  eigenValues, eigenVectors = eigsh(L,k=embedDim + 1,which='SM')
  return eigenVectors[:, 1:embedDim + 1 ]