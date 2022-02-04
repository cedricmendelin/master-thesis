# Generated with SMOP  0.41-beta
from libsmop import *
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m

    
@function
def vectordiffusionmap(vdm=None,*args,**kwargs):
    varargin = vectordiffusionmap.varargin
    nargin = vectordiffusionmap.nargin

    
    #  Vector diffusion map v0.1 
# 
# INPUT: 
#   vdm.data	: pxn matrix which represents n data points in R^p
#   vdm.epsilon	: \epsilon in the vDM
#   vdm.NN	: number of nearest neighbors
#   vdm.T	: diffusion time
#   vdm.delta	: parameter for the truncated VDM
#   vdm.origidx	: the index of the "origin point" you want to measure the affinity
#   vdm.symmetrize : symmetrize the graph
#   vdm.compact : compact support kernel (TODO)
    
    # (OPTION)
#   vdm.debug   : debug mode
#   vdm.parallel: parallel computation (TODO)
#   vdm.draw    : draw vector fields
#   vdm.drawNO  : draw the first drawNO vector fields
# 
# OUTPUT:
#   rslt.embeddim : the truncated VDM
#   rslt.embed    : the embedded data
#   rslt.vdd	 : the vector diffusion distance of each point to the origidx point
    
    # DEPENDENCE:
#   tstoolbox, localpca.m, (TESTING: KN_rankest.m)
    
    # by Hau-tieng Wu 2011-06-28
    
    if vdm.debug:
        fprintf('\n(DEBUG:vDM)\t\tStart to work on vector diffusion map\n')
    
    clc
    eigopt.isreal = copy(1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:34
    eigopt.issym = copy(1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:35
    eigopt.maxit = copy(3000)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:36
    eigopt.disp = copy(0)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:37
    pp,nn=size(vdm.data,nargout=2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:40
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if vdm.debug:
        fprintf('(DEBUG:vDM) Step 1: NN search and Data preparation. (For comuptational load issue)\n')
    
    X=(vdm.data).T
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:46
    atria=nn_prepare(X)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:46
    index,distance=nn_search(X,atria,concat([arange(1,nn)]).T,vdm.NN,- 1,0.0,nargout=2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:47
    if vdm.debug:
        fprintf(concat(['(DEBUG:vDM) NN=',num2str(vdm.NN),'\n']))
        fprintf(concat(['(DEBUG:vDM) minimal farest distance=',num2str(min(distance(arange(),end()))),'\n']))
        fprintf(concat(['(DEBUG:vDM) maximal farest distance=',num2str(max(distance(arange(),end()))),'\n']))
        fprintf(concat(['(DEBUG:vDM) median farest distance=',num2str(median(distance(arange(),end()))),'\n']))
        fprintf(concat(['(DEBUG:vDM) 1.5*sqrt(min farest distance)=',num2str(dot(1.5,sqrt(min(distance(arange(),end()))))),'.\n']))
        fprintf(concat(['(DEBUG:vDM) 1.5*sqrt(max farest distance)=',num2str(dot(1.5,sqrt(max(distance(arange(),end()))))),'.\n']))
        fprintf(concat(['(DEBUG:vDM) 1.5*sqrt(median farest distance)=',num2str(dot(1.5,sqrt(median(distance(arange(),end()))))),'.\n']))
    
    ## patchno is set to convert the NN information to the \sqrt{h} information
    patchno=dot(vdm.NN,ones(1,nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:60
    if vdm.debug == 1:
        fprintf('(DEBUG:vDM) the neighbors with kernel value less than exp(-5*1.5*1.5)=1.3e-5 are trimmed.\n')
    
    for ii in arange(1,nn).reshape(-1):
        patchno[ii]=length(find(distance(ii,arange()) < dot(1.5,sqrt(vdm.epsilon))))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:65
        distance[ii,find(distance(ii,arange()) > dot(1.5,sqrt(vdm.epsilon)))]=inf
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:66
    
    distance=distance(arange(),arange(1,max(patchno)))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:69
    index=index(arange(),arange(1,max(patchno)))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:70
    rslt.patchno = copy(patchno)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:71
    rslt.distance = copy(distance)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:71
    rslt.index = copy(index)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:71
    if vdm.debug == 1:
        if quantile(patchno,0.9) == vdm.NN:
            ## it means the NN is not big enough so that the decay of the kernel will not be enough and the error will be large
            fprintf('(DEBUG:vDM:WARNING) the NN should be chosen larger\n')
        fprintf(concat(['(DEBUG:vDM) the number of points with distance less then\n']))
        fprintf(concat(['           1.5*sqrt(epsilon)=',num2str(dot(1.5,sqrt(vdm.epsilon))),' is (min,max,median) = (',num2str(min(patchno)),',',num2str(max(patchno)),',',num2str(median(patchno)),')\n']))
        fprintf(concat(['(DEBUG:vDM) set NN to be ',num2str(max(patchno)),'\n']))
    
    NN=max(patchno)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:85
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if vdm.debug:
        fprintf('(DEBUG:vDM) Step 2: find a basis for each tangent plane by PCA\n')
    
    lpca.data = copy(vdm.data)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:92
    
    lpca.epsilonpca = copy(dot(vdm.epsilon,0.8))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:94
    lpca.NN = copy(NN)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:95
    lpca.index = copy(rslt.index)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:96
    lpca.distance = copy(rslt.distance)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:97
    lpca.patchno = copy(patchno)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:98
    
    lpca.KN = copy(0.9)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:100
    lpca.debug = copy(1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:101
    pcaBASIS,D=localpca(lpca,nargout=2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:103
    clear('lpca')
    clear('X')
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if vdm.debug:
        fprintf('(DEBUG:vDM) Step 3: deal with the reflection effect\t\t')
    
    REFLECTION=diag(ones(1,nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:109
    cback=0
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:111
    for ii in arange(1,nn).reshape(-1):
        for cc in arange(1,cback).reshape(-1):
            fprintf('\b')
        cback=fprintf('%4d',ii)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:113
        tmpSLICER=REFLECTION(ii,arange())
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:115
        for kk in arange(2,NN).reshape(-1):
            jj=index(ii,kk)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:118
            Ai=pcaBASIS(arange(),arange(1,D),ii)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:120
            Aj=pcaBASIS(arange(),arange(1,D),jj)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:121
            H=dot(Ai.T,Aj)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:122
            U,lambda_,V=svd(H,nargout=3)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:123
            X1=dot(V,U.T)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:124
            if det(X1) < 0:
                tmpSLICER[jj]=- 1
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:126
            else:
                tmpSLICER[jj]=1
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:126
        REFLECTION[ii,arange()]=tmpSLICER
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:129
    
    fprintf('\n')
    clear('tmpSLICER')
    clear('U')
    clear('V')
    clear('lambda')
    clear('X1')
    UR,lambdaR=eigs(REFLECTION,2,'lm',eigopt,nargout=2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:133
    
    for ii in arange(1,nn).reshape(-1):
        if UR(ii,1) < 0:
            frame=pcaBASIS(arange(),arange(),ii)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:137
            frame[arange(),1]=- frame(arange(),1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:137
            pcaBASIS[arange(),arange(),ii]=frame
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:137
    
    clear('UR')
    clear('lambdaR')
    clear('REFLECTION')
    clear('frame')
    rslt.pcaBASIS = copy(pcaBASIS)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:141
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    ## prepare for the graph
    if vdm.symmetrize == 1:
        if vdm.debug:
            fprintf('(DEBUG:vDM) Step 4': symmetrize the graph (very time consuming) \t\t')
        count=0
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:149
        cback=0
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:149
        for ii in arange(1,nn).reshape(-1):
            for cc in arange(1,cback).reshape(-1):
                fprintf('\b')
            cback=fprintf('%4d',ii)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:151
            for kk in arange(2,NN).reshape(-1):
                jj=index(ii,kk)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:153
                if logical_not(ismember(ii,index(jj,arange()))):
                    distance[ii,kk]=inf
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:154
                    count=count + 1
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:154
        if vdm.debug:
            fprintf(concat(['(DEBUG:vDM) removed entries=',num2str(dot(100,count) / (dot(size(distance,1),size(distance,2)))),'%%\n\n']))
    
    clear('count')
    
    Ac=zeros(dot(D,NN),dot(D,nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:163
    Dc=zeros(dot(D,nn),1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:164
    
    epsilon=vdm.epsilon
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:167
    if vdm.debug:
        fprintf('(DEBUG:vDM) Step 4: connection Laplacian operator\t\t')
    
    ## (TODO) \alpha power
	## work out "entries of A and D". For the sparse matrix purpose
    cback=0
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:173
    for ii in arange(1,nn).reshape(-1):
        for cc in arange(1,cback).reshape(-1):
            fprintf('\b')
        cback=fprintf('%4d',ii)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:175
        pii=sum(exp(dot(- 5,(distance(ii,arange()) ** 2 / epsilon))))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:177
        W2i=0
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:178
        for kk in arange(1,NN).reshape(-1):
            jj=index(ii,kk)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:182
            ## TODO: compact support kernel
            Kij=exp(dot(- 5,(distance(ii,kk) ** 2 / epsilon)))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:186
            pjj=sum(exp(dot(- 5,(distance(jj,arange()) ** 2 / epsilon))))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:187
            Kij2=Kij / (dot(pii,pjj))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:189
            W2i=W2i + Kij2
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:191
            Ai=pcaBASIS(arange(),arange(1,D),ii)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:193
            Aj=pcaBASIS(arange(),arange(1,D),jj)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:194
            H=dot(Ai.T,Aj)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:195
            U,lambda_,V=svd(H,nargout=3)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:196
            X1=dot(V,U.T)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:197
            Ac[arange(dot((kk - 1),D) + 1,dot(kk,D)),arange(dot((ii - 1),D) + 1,dot(ii,D))]=dot(X1,Kij2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:199
        Dc[arange(dot((ii - 1),D) + 1,dot(ii,D))]=W2i
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:203
    
    fprintf('\n')
    
    if isfield(vdm,'Nabla2'):
        if vdm.Nabla2:
            if vdm.debug:
                fprintf('(DEBUG:vDM) Get Connection Laplacian...\n')
            Cc=copy(Ac)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:215
            for ii in arange(1,nn).reshape(-1):
                Cc[arange(1,D),arange(dot((ii - 1),D) + 1,dot(ii,D))]=dot((1 / vdm.epsilon),Ac(arange(1,D),arange(dot((ii - 1),D) + 1,dot(ii,D)))) - diag(ones(D,1))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:218
    
    ## the following code are used to get the sparse matrix for either the connection Laplacian or its heat kernel
    I=zeros(dot(D,NN),dot(D,nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:224
    for ii in arange(1,dot(D,NN)).reshape(-1):
        I[ii,arange()]=arange(1,dot(D,nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:225
    
    J=zeros(dot(D,NN),dot(D,nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:227
    for ii in arange(1,nn).reshape(-1):
        H=zeros(dot(D,NN),1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:229
        for jj in arange(1,NN).reshape(-1):
            kk=index(ii,jj)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:231
            H[arange(dot((jj - 1),D) + 1,dot(jj,D))]=arange(dot((kk - 1),D) + 1,dot(kk,D))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:231
        for jj in arange(1,D).reshape(-1):
            J[arange(),dot((ii - 1),D) + jj]=H
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:232
    
    ## get A for the heat kernel
    sparseA=sparse(ravel(I),ravel(J),ravel(Ac),dot(D,nn),dot(D,nn),dot(dot(dot(D,NN),D),nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:237
    
    if isfield(vdm,'Nabla2'):
        if vdm.Nabla2:
            sparseC=sparse(ravel(I),ravel(J),ravel(Cc),dot(D,nn),dot(D,nn),dot(dot(dot(D,NN),D),nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:241
    
    clear('Ac')
    clear('Cc')
    clear('I')
    clear('J')
    clear('H')
    clear('REFLECTION')
    I=concat([arange(1,dot(D,nn),1)])
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:246
    
    sparseD=sparse(ravel(I),ravel(I),1.0 / Dc,dot(D,nn),dot(D,nn),dot(D,nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:249
    
    sparseS=dot(dot(sqrt(sparseD),sparseA),sqrt(sparseD))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:252
    
    sparseS=(sparseS + sparseS.T) / 2
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:255
    rslt.sparseS = copy(sparseS)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:256
    
    if isfield(vdm,'Nabla2'):
        if vdm.Nabla2:
            sparseC=dot(sparseD,sparseC)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:260
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if vdm.debug:
        fprintf('(DEBUG:vDM) step 5: find eigenvalues of the connection Laplacian operator\n')
    
    US,lambdaS=eigs(sparseS,vdm.lambdaNO,'lm',nargout=2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:266
    lambdaS=diag(lambdaS)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:267
    lambdaS,sortidx=sort(lambdaS,'descend',nargout=2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:268
    US=US(arange(),sortidx)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:269
    
    US=dot(sqrt(sparseD),US)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:270
    
    rslt.US = copy(US)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:272
    rslt.lambdaS = copy(lambdaS)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:273
    
    if isfield(vdm,'Nabla2'):
        if vdm.Nabla2:
            UC,lambdaC=eigs(sparseC,vdm.lambdaNO,'sm',nargout=2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:277
            lambdaC=dot(- 4,real(diag(lambdaC))) - (D - 1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:278
            rslt.lambdaC = copy(lambdaC)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:279
            rslt.UC = copy(UC)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:280
    
    clear('UC')
    clear('lambdaC')
    clear('sparseC')
    if vdm.debug:
        subplot(2,1,1)
        bar(lambdaS)
        set(gca,'fontsize',vdm.fontsize)
        title('(DEBUG) the first 200 eigenvalues\newline(note the scale)')
        axis('tight')
        subplot(2,1,2)
        bar((lambdaS / lambdaS(1)) ** (dot(2,vdm.T)))
        set(gca,'fontsize',vdm.fontsize)
        title('(DEBUG) the diffusion behavior of the eigenvalues\newline(note the scale)')
        axis('tight')
    
    
    ## TODO: setup some parameters for special manifolds, for example, S^n
    
    # lambdaS(1:6)=lambdaS(1); ## if sphere, set the first 6 eigenvalues the same artificially
    
    if vdm.debug:
        fprintf(concat(['(DEBUG:vDM) The diffusion time T=',num2str(vdm.T),', and the threshold is ',num2str(vdm.delta),'\n']))
    
    dimidx=find((lambdaS / lambdaS(1)) ** (dot(2,vdm.T)) > vdm.delta)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:305
    dimno=length(dimidx)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:306
    rslt.embeddim = copy(dot(dimno,(dimno + 1)) / 2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:307
    fprintf(concat(['\t\tVector diffusion map will embed the dataset into ',num2str(rslt.embeddim),'-dim Euclidean space\n\n']))
    if logical_and(vdm.debug,rslt.embeddim) > 500:
        fprintf('(DEBUG:vDM:WARNING) the embedding dimension might be too big, really continue?\n')
        pause
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    ##
	## The following code is written to save space. Since the VDM
	## might embed the data to much higher dimensional space,
	## be careful if you want to save all the embedded data
	## To save space, we only save the vector diffusion distance
	##
    if isfield(vdm,'vdd'):
        if vdm.vdd:
            if vdm.debug:
                fprintf('(DEBUG:vDM) step 6: VDM and VDD\t\t')
            x0pt=zeros(rslt.embeddim,1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:327
            ss=1
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:329
            for ii in arange(1,dimno).reshape(-1):
                for jj in arange(ii,dimno).reshape(-1):
                    if ii != jj:
                        x0pt[ss]=dot(dot(dot(sqrt(2),((dot(lambdaS(ii),lambdaS(jj))) ** (vdm.T))),US(arange(dot((vdm.origidx - 1),D) + 1,dot(vdm.origidx,D)),ii).T),US(arange(dot((vdm.origidx - 1),D) + 1,dot(vdm.origidx,D)),jj))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:333
                    else:
                        x0pt[ss]=dot(dot(((dot(lambdaS(ii),lambdaS(jj))) ** (vdm.T)),US(arange(dot((vdm.origidx - 1),D) + 1,dot(vdm.origidx,D)),ii).T),US(arange(dot((vdm.origidx - 1),D) + 1,dot(vdm.origidx,D)),jj))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:337
                    ss=ss + 1
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:342
            rslt.vdd = copy(zeros(1,nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:347
            xipt=zeros(size(x0pt))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:349
            tic
            cback=0
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:350
            for qq in arange(1,nn).reshape(-1):
                for cc in arange(1,cback).reshape(-1):
                    fprintf('\b')
                cback=fprintf('%4d',qq)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:352
                ss=1
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:353
                for ii in arange(1,dimno).reshape(-1):
                    for jj in arange(ii,dimno).reshape(-1):
                        if ii != jj:
                            xipt[ss]=dot(dot(dot(sqrt(2),((dot(lambdaS(ii),lambdaS(jj))) ** (vdm.T))),US(arange(dot((qq - 1),D) + 1,dot(qq,D)),ii).T),US(arange(dot((qq - 1),D) + 1,dot(qq,D)),jj))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:357
                        else:
                            xipt[ss]=dot(dot(((dot(lambdaS(ii),lambdaS(jj))) ** (vdm.T)),US(arange(dot((qq - 1),D) + 1,dot(qq,D)),ii).T),US(arange(dot((qq - 1),D) + 1,dot(qq,D)),jj))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:361
                        ss=ss + 1
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:365
                rslt.vdd[qq]=norm(xipt - x0pt)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:368
            fprintf('\n')
            toc
            if vdm.debug:
                if pp == 3:
                    figure
                    scatter3(vdm.data(1,arange()),vdm.data(2,arange()),vdm.data(3,arange()),10,ravel(rslt.vdd),'fill')
                    hold('on')
                    plot3(vdm.data(1,vdm.origidx),vdm.data(2,vdm.origidx),vdm.data(3,vdm.origidx),'. red','markersize',30)
                    view(concat([30,90]))
                    colorbar
                    set(gca,'fontsize',vdm.fontsize)
                    axis('tight')
                    axis('off')
                    title('The vector diffusion distance from the red point')
                else:
                    if pp == 2:
                        figure
                        scatter(vdm.data(1,arange()),vdm.data(2,arange()),10,ravel(rslt.vdd),'fill')
                        hold('on')
                        plot(vdm.data(1,vdm.origidx),vdm.data(2,vdm.origidx),'. red','markersize',30)
                        colorbar
                        set(gca,'fontsize',vdm.fontsize)
                        axis('tight')
                        axis('off')
                        title('The vector diffusion distance from the red point')
                    else:
                        if pp == 1:
                            figure
                            plot(vdm.data,rslt.vdd,'.')
                            set(gca,'fontsize',vdm.fontsize)
                            hold('on')
                            plot(vdm.data(vdm.origidx),rslt.vdd(vdm.origidx),'. red','markersize',30)
                            title('The vector diffusion distance from the red point')
                        else:
                            fprintf('Sorry, but I don't know how to visualize the high dimensional manifold\n')
        ###################################################################
# draw vector fields
        if vdm.draw:
            if pp == logical_and(3,D) == 2:
                trueevf=zeros(3,nn,vdm.drawno)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:404
                for qq in arange(1,vdm.drawno).reshape(-1):
                    figure
                    evf=reshape(US(arange(),qq),D,nn)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:406
                    evf3d=zeros(3,nn)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:407
                    for ii in arange(1,nn).reshape(-1):
                        evf3d[arange(),ii]=dot(pcaBASIS(arange(),arange(),ii),evf(arange(),ii))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:410
                    trueevf[arange(),arange(),qq]=evf3d
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:413
                    IDXX=concat([arange(1,nn)])
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:414
                    ## the following is for detailed view purpose
           # XYZ=zeros(1,6);
    	   # vfnorm=zeros(1,nn);
    	   # for ii=1:nn; vfnorm(ii)=norm(evf3d(:,ii)); end
    	   # [aa,bb]=sort(vfnorm);
           # XYZ(ii)=bb(1);	## find the smallest norm
    	   # OOO=zeros(1,nn);	## get a neighborhood of bb(1)
    	   # for ii=1:nn; OOO(ii)=norm(vdm.data(:,ii)-vdm.data(:,bb(1))); end
    	   # IDXX=find(OOO<=1);
                    quiver3(vdm.data(1,IDXX(arange(1,end(),10))),vdm.data(2,IDXX(arange(1,end(),10))),vdm.data(3,IDXX(arange(1,end(),10))),evf3d(1,IDXX(arange(1,end(),10))),evf3d(2,IDXX(arange(1,end(),10))),evf3d(3,IDXX(arange(1,end(),10))),1,'linewidth',2)
                    hold('on')
                    plot3(vdm.data(1,IDXX(arange(1,end(),10))),vdm.data(2,IDXX(arange(1,end(),10))),vdm.data(3,IDXX(arange(1,end(),10))),'. red','markersize',10)
                    set(gca,'fontsize',vdm.fontsize)
                    axis('tight')
                    axis('equal')
                    title(concat(['The ',num2str(qq),'-th(blue) eigenvector field']))
            else:
                if pp == logical_and(2,D) == 2:
                    IDX=concat([arange(1,nn)])
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:438
                    IDX=reshape(IDX,sqrt(nn),sqrt(nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:439
                    IDX=IDX(arange(1,end(),2),arange(1,end(),2))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:440
                    IDX=ravel(IDX)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:441
                    trueevf=zeros(2,nn,vdm.drawno)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:442
                    for qq in arange(1,vdm.drawno).reshape(-1):
                        figure
                        evf=reshape(US(arange(),qq),D,nn)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:444
                        pcaBASIS=pcaBASIS(arange(),arange(1,D),arange())
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:445
                        evf2d=zeros(2,no)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:446
                        for ii in arange(1,nn).reshape(-1):
                            evf2d[arange(),ii]=dot(pcaBASIS(arange(),arange(),ii),evf(arange(),ii))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:448
                        trueevf[arange(),arange(),qq]=evf2d
# C:\Users\CedricMendelin\Downloads\VDM\VDM\vectordiffusionmap.m:449
                        quiver(vdm.data(1,IDX),vdm.data(2,IDX),evf2d(1,IDX),evf2d(2,IDX),0.5,'linewidth',2,'color','black')
                        hold('on')
                        plot(vdm.data(1,IDX),vdm.data(2,IDX),'. red','markersize',10)
                        set(gca,'fontsize',vdm.fontsize)
                        title(concat(['The ',num2str(qq),'-th(blue) eigenvector field']))
                        grid('on')
                else:
                    if pp == logical_and(1,D) == 1:
                        ## TODO
                        pass
                    else:
                        if pp == logical_and(2,D) == 1:
                            ## TODO
                            pass
                        else:
                            if pp == logical_and(3,D) == 1:
                                ## TODO
                                pass
                            else:
                                fprintf('Sorry, but there is no way to draw high dimensional manifold\n')
    