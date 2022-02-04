# Generated with SMOP  0.41-beta
from libsmop import *
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m

    
@function
def dim_localpca(lpca=None,*args,**kwargs):
    varargin = dim_localpca.varargin
    nargin = dim_localpca.nargin

    
    # dimension estimation by local PCA ver 0.3
    
    # INPUT:
#   lpca.data      = pxn matrix, n points in R^p
#   lpca.epsilonpca   = kernel parameter (epsilonpca_PCA)
#   lpca.NN        = the number of nearest neighbors chosen in the algorithm
#		     should be large enough.
#   lpca.index	   = index determined by tstoolbox
#   lpca.distance  = distance determined by tstoolbox
# (OPTION)
#   lpca.KN	   = using KN algorithm or not. If =1, then use KN, If<1
#		     then lpca.KN is used as the threshold_gamma
#   lpca.debug     = debug mode.
    
    # OUTPUT:
#   pcaBASIS	= eigenfunctions of Laplace-Beltrami
#   estdim	= estimated dimension
# 
# DEPENDENCY:
#   tstoolbox, KN_rankest
    
    # by Hau-tieng Wu 2011-06-20 (hauwu@math.princeton.edu)
    
    if lpca.debug == 1:
        fprintf(concat(['(DEBUG:lpca) NN=',num2str(lpca.NN),'; kernel parameter=',num2str(lpca.epsilonpca),'\n']))
    
    eigopt.isreal = copy(1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:31
    eigopt.issym = copy(1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:32
    eigopt.maxit = copy(3000)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:33
    eigopt.disp = copy(0)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:34
    pp,nn=size(lpca.data,nargout=2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:36
    if lpca.KN < logical_and(1,lpca.KN) > 0:
        threshold_gamma=lpca.KN
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:39
        if lpca.debug == 1:
            fprintf('(DEBUG:lpca) Use threshold_gamma to determine the local dimension\n')
    else:
        if lpca.KN > logical_or(1,lpca.KN) < 0:
            error('threshold_gamma should be inside [0,1]')
        else:
            if lpca.debug == 1:
                fprintf('(DEBUG:lpca) Use KN to determine the local dimension\n')
    
    
    if lpca.debug == 1:
        if pp > nn:
            fprintf('(DEBUG:lpca) It is better to do dimension reduction, for example, PCA, to reduce the ambient space dimention\n')
    
    pcaBASIS=zeros(pp,pp,nn)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:58
    ldim=ones(nn,1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:59
    data=lpca.data
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:61
    distance=lpca.distance
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:62
    index=lpca.index
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:63
    NN=lpca.NN
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:64
    patchno=lpca.patchno
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:65
    if lpca.KN == 1:
        for ii in arange(1,nn).reshape(-1):
            KNpca.data = copy(data(arange(),index(ii,arange(1,patchno(ii)))))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:70
            KNpca.debug = copy(1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:71
            UPC,lambdaPC,NOPC,sigmahat2=KN_PCA(KNpca,nargout=4)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:72
            ldim[ii]=NOPC
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:73
            pcaBASIS[arange(),arange(),ii]=UPC
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:74
    else:
        ## spectral PCA
        Q=exp(dot(- 5,(distance ** 2.0 / lpca.epsilonpca)))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:79
        Q=Q.T
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:79
        Q=ravel(Q)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:79
        I=zeros(NN,nn)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:81
        for k in arange(1,NN).reshape(-1):
            I[k,arange()]=arange(1,nn)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:81
        I=ravel(I)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:81
        J=index.T
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:82
        J=ravel(J)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:82
        W=sparse(I,J,ravel(Q),nn,nn,dot(nn,NN))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:84
        D=sum(W)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:85
        D=ravel(D)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:85
        for ii in arange(1,nn).reshape(-1):
            CENTER=data(arange(),ii)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:89
            Aii=data(arange(),index(ii,arange(2,patchno(ii)))) - repmat(CENTER,1,patchno(ii) - 1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:90
            Dii=diag(W(ii,index(ii,arange(2,patchno(ii)))))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:91
            Cor=dot(Aii,Dii) / sqrt(D(ii))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:92
            U,lambda_,UT=svd(Cor,nargout=3)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:93
            if size(lambda_,1) != 1:
                lambda_=diag(lambda_)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:95
            else:
                lambda_=copy(lambda_)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:95
            totalenergy=sum(lambda_)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:96
            ldim[ii]=1
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:97
            energy=0
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:97
            while true:

                energy=energy + lambda_(ldim(ii))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:100
                if energy / totalenergy > threshold_gamma:
                    break
                ldim[ii]=ldim(ii) + 1
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:102

            pcaBASIS[arange(),arange(),ii]=U(arange(),arange())
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:104
    
    estdim=median(ldim)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:111
    if lpca.debug:
        fprintf(concat(['(DEBUG:lpca)\t estimated dimension=',num2str(estdim),'\n']))
    
    pcaBASIS=pcaBASIS(arange(),arange(1,estdim),arange())
# C:\Users\CedricMendelin\Downloads\VDM\VDM\localpca.m:114
    if lpca.debug == 10:
        hist(dim,100)
        axis('tight')
        axis(concat([0,10,- inf,inf]))
        set(gca,'fontsize',10)
        title('(DEBUG:lpca) the histogram of the estimated dimension at each point')
    