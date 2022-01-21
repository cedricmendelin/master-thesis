import os.path
import numpy as np
from sklearn.neighbors import NearestNeighbors
# cur_dir = os.path.dirname(os.path.realpath(__file__))
# par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
# sys.path.append('%s/software/' % cur_dir)

from software.REGAL.alignments import get_embedding_similarities
from scipy import sparse
from scipy.sparse.linalg import eigsh


import numpy as np 
from scipy import sparse




class GAlign:
    def __init__(self, emb_method, G1=None, G2=None,emb1=None,emb2=None,ndim=3,max_layer = 2):
        if emb_method=='Lap':
            self.G1 = G1
            self.G2 = G2
            self.n=G1.shape[0]
            self.ndim=ndim
            self.get_emb(ndim=ndim)
        elif emb_method=='xnetmf':
            self.G1 = G1
            self.G2 = G2
            self.n=G1.shape[0]
            self.get_emb_xnetmf(max_layer=max_layer)
        else:
            self.emb1=emb1
            self.emb2=emb2
            self.n=emb1.shape[0]
    def emb_L(self,G,ndim=3):
        A_nl=G
        A_knn=0.5*(A_nl+A_nl.T)
        L = np.diag(A_knn.sum(axis=1)) - A_knn
        L=sparse.csr_matrix(L)
        eigenValues, eigenVectors=eigsh(L,k=ndim+1,which='SM')
        idx = np.argsort(eigenValues)
        emb=eigenVectors[:, idx[1]].reshape(-1,1)
        for i in range(ndim-1):
            emb=np.concatenate([emb,eigenVectors[:, idx[i+2]].reshape(-1,1)],axis=1)
        row_norm = np.sqrt((emb**2).sum(axis=1))
        emb=emb/row_norm[:, np.newaxis]
        return emb
    def get_emb(self,ndim=3):
        self.emb1=self.emb_L(self.G1,ndim=ndim)
        self.emb2=self.emb_L(self.G2,ndim=ndim)

    def get_emb_xnetmf(self,max_layer=3):
        from software.REGAL.xnetmf import get_representations
        from software.REGAL.config import RepMethod,Graph
        from software.REGAL.alignments import get_embeddings
        G1 = sparse.csr_matrix(self.G1)
        G2 = sparse.csr_matrix(self.G2)
        comb = sparse.block_diag([G1,G2])
        graph = Graph(adj = comb.tocsr())
        rep_method = RepMethod(max_layer = max_layer)
        representations = get_representations(graph, rep_method)
        self.emb1,self.emb2=get_embeddings(representations)






    def get_sim(self):
        self.sim=get_embedding_similarities(self.emb1, self.emb2, sim_measure = "euclidean", num_top = None)
        return self.sim
    def get_align(self,q=True):
        sim=self.get_sim()
        candidate=np.arange(self.n)
        align_results=[]
        for i in range(self.n):
            sim_i=sim[i,candidate]
            idx=np.argmax(sim_i)
            nidx=candidate[idx]
            align_results.append(nidx)
            if not q:
                print(candidate,nidx)
            candidate=np.delete(candidate,idx)
        return align_results