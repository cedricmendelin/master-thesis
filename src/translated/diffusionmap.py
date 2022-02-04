# Generated with SMOP  0.41-beta
from libsmop import *
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m

    
@function
def diffusionmap(dm=None,*args,**kwargs):
    varargin = diffusionmap.varargin
    nargin = diffusionmap.nargin

    
    # Diffusion map ver 0.3
# Try example_dm.m to see example
    
    # INPUT:
#   dm.data 	 = pxn matrix, n points in R^p
#   dm.epsilon   = kernel parameter
#   dm.NN    	 = the number of nearest neighbors chosen in the algorithm
#   dm.T     	 = diffusion time
#   dm.delta 	 = truncation threshold
# (OPTIONS)
#   dm.parallel   = use parallel computataion
#   dm.symmetrize = symmetrize the graph
#   dm.compact    = compact support kernel
#   dm.debug      = debug message
#   dm.embedding  = embedding according to the truncation
    
    # OUTPUT:
#   rslt.Delta	  = Laplace-Beltrami operator
#   rslt.UDelta	  = eigenfunctions of Laplace-Beltrami
#   rslt.lambdaDelta = eigenvalues of Laplace-Beltrami
#   rslt.embeddim = embedded dimension
#   rslt.embedded = embedded data points
# 
# DEPENDENCY:
#   tstoolbox
    
    # by Hau-tieng Wu 2011-06-20 (hauwu@math.princeton.edu)
#
    if dm.debug:
        fprintf('\n(DEBUG:DM)\t\tStart to work on Diffusion Map\n')
    
    if dm.cleanup:
        clc
        close_('all')
    
    if dm.parallel:
        matlabpool('open')
    
    data=dm.data
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:35
    NN=dm.NN
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:35
    epsilon=dm.epsilon
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:35
    eigopt.isreal = copy(1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:37
    eigopt.issym = copy(1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:38
    eigopt.maxit = copy(3000)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:39
    eigopt.disp = copy(0)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:40
    pp,nn=size(data,nargout=2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:43
    
    if NN < pp + 1:
        if dm.parallel == 1:
            matlabpool('close')
        error('*** ERROR: choose NN>p')
    
    if logical_or(logical_or(logical_not(isfield(dm,'distance')),logical_not(isfield(dm,'index'))),logical_not(isfield(dm,'patchno'))):
        X=data.T
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:54
        atria=nn_prepare(X)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:54
        index,distance=nn_search(X,atria,concat([arange(1,nn)]).T,NN,- 1,0.0,nargout=2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:55
        clear('X')
        if dm.debug:
            fprintf(concat(['(DEBUG:DM) NN=',num2str(dm.NN),'\n']))
            fprintf(concat(['(DEBUG:DM) minimal farest distance=',num2str(min(distance(arange(),end()))),'\n']))
            fprintf(concat(['(DEBUG:DM) maximal farest distance=',num2str(max(distance(arange(),end()))),'\n']))
            fprintf(concat(['(DEBUG:DM) median farest distance=',num2str(median(distance(arange(),end()))),'\n']))
            fprintf(concat(['(DEBUG:DM) 1.5*sqrt(min farest distance)=',num2str(dot(1.5,sqrt(min(distance(arange(),end()))))),'.\n']))
            fprintf(concat(['(DEBUG:DM) 1.5*sqrt(max farest distance)=',num2str(dot(1.5,sqrt(max(distance(arange(),end()))))),'.\n']))
            fprintf(concat(['(DEBUG:DM) 1.5*sqrt(median farest distance)=',num2str(dot(1.5,sqrt(median(distance(arange(),end()))))),'.\n']))
        ## patchno is set to convert the NN info to the \sqrt{h} info
        patchno=dot(dm.NN,ones(1,nn))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:69
        if dm.debug == 1:
            fprintf('(DEBUG:DM) the neighbors with kernel value less than exp(-5*1.5*1.5)=1.3e-5 are trimmed.\n')
        for ii in arange(1,nn).reshape(-1):
            patchno[ii]=length(find(distance(ii,arange()) < dot(1.5,sqrt(dm.epsilon))))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:74
            distance[ii,find(distance(ii,arange()) > dot(1.5,sqrt(dm.epsilon)))]=inf
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:75
        distance=distance(arange(),arange(1,max(patchno)))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:78
        index=index(arange(),arange(1,max(patchno)))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:79
        rslt.patchno = copy(patchno)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:80
        rslt.distance = copy(distance)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:80
        rslt.index = copy(index)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:80
        if dm.debug == 1:
            if quantile(patchno,0.9) == dm.NN:
                ## it means the NN is not big enough so that the decay of the kernel will not be enough and the error will be large
                fprintf('(DEBUG:DM:WARNING) the NN should be chosen larger\n')
            fprintf(concat(['(DEBUG:DM) the number of points with distance less then\n']))
            fprintf(concat(['           1.5*sqrt(epsilon)=',num2str(dot(1.5,sqrt(dm.epsilon))),' is (min,max,median) = (',num2str(min(patchno)),',',num2str(max(patchno)),',',num2str(median(patchno)),')\n']))
            fprintf(concat(['(DEBUG:DM) set NN to be ',num2str(max(patchno)),'\n']))
        NN=max(patchno)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:94
    else:
        index=dm.index
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:97
        distance=dm.distance
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:97
        patchno=dm.patchno
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:97
    
    if dm.symmetrize == 1:
        fprintf('START: symmetrize the graph (will be slow).\n')
        count=0
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:104
        cc=0
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:104
        for ii in arange(1,no).reshape(-1):
            if mod(ii,no / 100) == 0:
                for kk in arange(1,cc).reshape(-1):
                    fprintf('\b')
                cc=fprintf('%3.2f%%',dot(100,ii) / no)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:108
            for kk in arange(2,NN).reshape(-1):
                jj=index(ii,kk)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:112
                if logical_not(ismember(ii,index(jj,arange()))):
                    distance[ii,kk]=inf
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:115
                    count=count + 1
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:116
        fprintf('\nFINISH: symmetrize the graph.\n')
        if dm.debug == 1:
            fprintf(concat(['(DEBUG:DM) removed entries=',num2str(dot(100,count) / (dot(size(distance,1),size(distance,2)))),'%%\n\n']))
    
    Q=exp(dot(- 5,(distance ** 2.0 / epsilon)))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:135
    Q=Q.T
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:135
    Q=ravel(Q)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:135
    I=zeros(NN,nn)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:137
    for k in arange(1,NN).reshape(-1):
        I[k,arange()]=arange(1,nn)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:138
    
    I=ravel(I)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:139
    J=index.T
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:141
    J=ravel(J)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:141
    W=sparse(I,J,ravel(Q),nn,nn,dot(nn,NN))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:143
    W=multiply(W,W.T)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:145
    
    D=sum(W)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:146
    D=ravel(D)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:146
    D1=sparse(arange(1,length(D)),arange(1,length(D)),1.0 / D)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:147
    W1=dot(dot(D1,W),D1)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:148
    
    D=sqrt(sum(W1))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:149
    D=ravel(D)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:149
    D2=sparse(arange(1,length(D)),arange(1,length(D)),1.0 / D)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:150
    W2=dot(dot(D2,W1),D2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:151
    
    W2=(W2 + W2.T) / 2
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:152
    
    if dm.debug:
        tic
        fprintf('(DEBUG:DM) Calculate eigenfunctions/eigenvalues...')
    
    UD,lambdaUD,UDT=svds(W2,80,'L',nargout=3)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:156
    if dm.debug:
        fprintf('DONE! \n')
        toc
    
    UD=dot(D2,UD)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:159
    lambdaUD=diag(lambdaUD)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:159
    lambdaUD,sortidx=sort(lambdaUD,'descend',nargout=2)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:160
    UD=UD(arange(),sortidx)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:161
    rslt.UL = copy(UD)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:163
    rslt.lambdaL = copy(lambdaUD)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:164
    keyboard
    if dm.embedding:
        dimidx=find((lambdaUD / lambdaUD(2)) ** dm.T > dm.delta)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:169
        embeddim=length(dimidx)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:170
        if dm.debug:
            fprintf(concat(['(DEBUG:DM) Diffusion map will embed the data to ',num2str(embeddim),'-dim Euclidean space\n']))
        rslt.embeddim = copy(embeddim)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:172
        embedded=dot(UD(arange(),arange(2,embeddim)),diag(lambdaUD(arange(2,embeddim)) ** dm.T))
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:174
        embedded=embedded.T
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:175
        rslt.embedded = copy(embedded)
# C:\Users\CedricMendelin\Downloads\VDM\VDM\diffusionmap.m:176
    
    if dm.display:
        figure
        bar(lambdaUD)
        axis('tight')
        axis(concat([- inf,inf,0.95,1]))
        figure
        plot3(UD(arange(),4),UD(arange(),2),UD(arange(),3),'.')
        axis('tight')
        figure
        plot(UD(arange(),2),UD(arange(),3),'.')
        axis('tight')
    
    if dm.parallel == 1:
        matlabpool('close')
    