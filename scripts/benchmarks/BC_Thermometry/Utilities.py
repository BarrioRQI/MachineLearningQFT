import numpy as np
from math import floor
from numpy.linalg import inv
from scipy.linalg import expm, sqrtm, tanhm

###############################################################################
############################# DEFINITIONS #####################################
###############################################################################

#Order of appearance in datagenerator

def InitializeProbeState(State='Ground'):
    if State == 'Ground':                                   # Initialize probe to ground state - Gaussian
        return np.array([[1,0],[0,1]])
    else:
        print("Error: Pick Valid Inital State")
    return np.nan()

############################################################################
##################### DEFINING HAMILTONIANS ################################
############################################################################

def ComputeHams(wD,mcc,lam,LatLen,sigma,B,D):
    N_Env = LatLen
    #print('Defining Probe Local Hamiltonian')
    HS = wD * np.eye(2)     #In gaussian we define F with H = 1/2 R F R^T
    #print('Defining Environment Local Hamiltonian')
    HE0 = np.zeros((2*N_Env,2*N_Env))
    HE0[0:N_Env,0:N_Env] = (mcc+2/mcc) * np.eye(N_Env)
    HE0[N_Env:2*N_Env,N_Env:2*N_Env] = mcc * np.eye(N_Env)
        
    #print('Defining Environment Internal Hamiltonian for Bulk')
    AdjBulk = SquareLatticeAdjList(LatLen)
    HEint = (1/mcc) * EnvIntHam(AdjBulk)
    
    if B == 1:
        HE_dynamic = HE0 + HEint
        HE_thermal = HE0 + HEint
    elif B == 2:
        HEint[0,1] = 0
        HEint[1,0] = 0        
        HE_dynamic = HE0 + HEint
        HE_thermal = HE0 + HEint
    elif B == 3:
        HE_dynamic = HE0 + HEint
        HEint[0,1] = 0
        HEint[1,0] = 0        
        HE_thermal = HE0 + HEint
    
    #print('Defining Probe - Environment Hamiltonian')
    if D > LatLen: print('Error: D too large')

    HSA  = lam * SetupSAHam(D,sigma,N_Env)       # Define the system ancilla coupling    
    Ham = TotalHam(HS,HSA,HE_dynamic)
    return Ham, HE_thermal

### Creates the adjacency list for a d-dimensional square lattice of size L  ###
def SquareLatticeAdjList(L,d=1,IncludeBulk=True, IncludePeriodic = False,JustFilm=False,CutFilm=False):      
    def Coord(i,L,d):
        if d == 0:
            return []
        return  [i%L] + Coord(int(i/L), L, d-1)

    def Index(coor,L):
        return sum(coor[r]*L**r for r in range(len(coor)))
    
    adjLists = []                           # Initialize the adjacency list
    for index in range(L**d):             # For each site in the lattice
        n = []                              # Initialize its list of neighbors         
        coor = Coord(index,L,d)
        for k in range(2*d):                           
            dirr = [0]*d
            dirr[int(k/2)] = (-1)**(k % 2)
            coor_n = [coor[r] + dirr[r] for r in range(len(coor))]
            
            if JustFilm == True:
                if k == 0 and coor_n[int(k/2)] == 1 or k == 1 and coor_n[int(k/2)]==0:
                    n.append(Index(coor_n,L))            

            #Checks if we past the forward edge 
            elif coor_n[int(k/2)] == L:
                # if so check if we are including the boundary
                if IncludePeriodic == True:
                    coor_n[int(k/2)] = 0      # if so, wrap around the lattice
                    n.append(Index(coor_n,L)) # and append the neighbor to our list

            #Checks if we past the backwards edge
            elif coor_n[int(k/2)] == -1:
                #if so check if we are including the boundary
                if IncludePeriodic == True:
                    coor_n[int(k/2)] = L-1      # if so, wrap around the lattice
                    n.append(Index(coor_n,L))   # and append the neighbor to our list  

            elif CutFilm == True and (k == 0 and coor_n[int(k/2)] == 1 or k == 1 and coor_n[int(k/2)]==0):                                # Otherwise proceed naively
                continue

            #if both those tests have failed we must be in the bulk
            else:   
                #Check if we are including the bulk
                if IncludeBulk == True:
                    n.append(Index(coor_n,L))   # if so, add the naive neighbor to the list

        adjLists.append(n)                      # Append the list of neigbors to the adjacency list
    return adjLists


### Defines Environment Hamiltonian based on its geometry and coupling type ###
### HEint = sum_<ij> sum_{r,t} J[r,t] S_{i,r} * S_{j,t} 
def EnvIntHam(adjList):    
    N_E = len(adjList)                      # Identify number of spins in environment
    F = np.zeros((2*N_E ,2*N_E));        
    for n_i in range(len(adjList)):                 # For each site i 
        for n_j in adjList[n_i]:                    # For each neighboring site j
            F[n_i,n_j] = -1;  
    return F                               # Updates xx part of the matrix    return F

def HEBound(B,AdjBound):
    if B == 1:
        HEint = 1 * EnvIntHam(AdjBound)
    elif B == 2:
        HEint = 0 * EnvIntHam(AdjBound)
    else:
        print('Error: Pick valid boundary case')
    return HEint

def SetupSAHam(n_A,sigma,N_E):    
    n_A = n_A - 1    
    F= np.zeros((2*N_E +2,2*N_E +2));
    for x in range(N_E):
        weight = np.exp(-(x-n_A)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
        F[N_E,x] = weight;            # Updates xx part of the matrix
        F[x,N_E]  = weight;    
    return F       
    
### Picks which coupling case and constructs the total Hamiltonian ###
def TotalHam(HS,HSA,HE):
    HS1 = directsum(np.zeros(HE.shape),HS)
    HE1 = directsum(HE, np.zeros(HS.shape))
    F = HSA + HS1 + HE1
    return F

def directsum(A,B):
    m1= floor(A.shape[0]/2)
    m2= floor(B.shape[0]/2)
    
    F= np.zeros((2*(m1+m2),2*(m1+m2)))
    F[0:m1,0:m1]=A[0:m1,0:m1]
    F[m1:m1+m2,m1:m1+m2]=B[0:m2,0:m2]
    F[m1+m2:2*m1+m2,m1+m2:2*m1+m2]=A[m1:2*m1,m1:2*m1]
    F[2*m1+m2:2*(m1+m2),2*m1+m2:2*(m1+m2)]=B[m2:2*m2,m2:2*m2]
    return F

### Computes unitaries at k time points between Tmin and Tmix under Ham ###
def DefProjList(Ham,N_t,Tmin,Tmax):
    dim = Ham.shape[0]
    if Tmin == 0: Tmin = Tmax/N_t
    Tlist=np.linspace(Tmin,Tmax,N_t)              # Divide evolution period into k pieces
    dT = Tlist[1]-Tlist[0]                        # Compute time step
    
    Ulist = np.zeros((N_t,dim,dim)) # Initialize a list of unitaries
    UAdlist = np.zeros((N_t,dim,dim)) # Initialize a list of unitaries
#    print('Ham',Ham)
#    print('multOmega(Ham)',multOmega(Ham))
#    print('np.asarray(multOmega(Ham)*dT)',np.asarray(multOmega(Ham)*dT))
    
    Ustep = expm(np.asarray(multOmega(Ham)*dT))
    Ulist[0] = expm(np.asarray(multOmega(Ham)*Tmin))
#   Ustep = Sexp(Ham,dT)
#   Ulist[0] = Sexp(Ham,Tmin)           # Compute initial unitary
    UAdlist[0] = np.transpose(Ulist[0])
    for k in range(1,N_t):
        Ulist[k] = Ustep @ Ulist[k-1]
        UAdlist[k] = np.transpose(Ulist[k])        

    m = int(dim/2)
    P0 = np.zeros((2*m,2*m))
    P1 = np.zeros((2*m,2*m))
    P2 = np.zeros((2*m,2*m))
    P0[m-1,m-1] = 1
    P1[m-1,2*m-1] = -1/2
    P1[2*m-1,m-1] = -1/2
    P2[2*m-1,2*m-1] = 1
    P1 += (P0+P2)/2                    
    Proj0 = [P0,P1,P2]
    ProjList = np.zeros((N_t,3,2*m,2*m))

    for n in range(N_t):
        for r in range(3):
            ProjList[n,r] = UAdlist[n] @ Proj0[r] @ Ulist[n]

    return ProjList

def Sexp(Ham,t,K=1000):
    m = int(Ham.shape[0]/2)

    Q = np.zeros(Ham.shape)
    Q[0:m,0:m] = Ham[0:m,0:m]
    Q = multOmega(Q)

    P = np.zeros(Ham.shape)
    P[m:2*m,m:2*m] = Ham[m:2*m,m:2*m]
    P = multOmega(P)

#    w = abs(Ham[0,0])
#    g = abs(Ham[0,1])
#    mp = 10**(-20)
#    K=1+int(((w+g)**5 * t**5/mp)**(1/4.5))
#    print(K)
    
    U = np.eye(2*m)
    dQ = Q*t/K
    dP = P*t/K
    
    r = 2 - 2**(1/3)
    c1 = 1/(2*r)
    d1 = 1/r
    c2 = (r-1)/(2*r)
    d2 = (r-2)/r
    c3 = (r-1)/(2*r)
    d3 = 1/r
    c4 = 1/(2*r)    
    
    for k in range(K):
        U = (np.eye(2*m)+c1*dQ) @ U
        U = (np.eye(2*m)+d1*dP) @ U
        U = (np.eye(2*m)+c2*dQ) @ U
        U = (np.eye(2*m)+d2*dP) @ U
        U = (np.eye(2*m)+c3*dQ) @ U
        U = (np.eye(2*m)+d3*dP) @ U
        U = (np.eye(2*m)+c4*dQ) @ U
    return U    

def multOmega(sigma):   
    p = int(round(len(sigma)/2));
    res =np.matrix([[0 + 0j for i in range(0,2*p)] for j in range(0,2*p)])
    res[0:p, 0:p]= sigma[p:2*p, 0:p];
    res[0:p, p:2*p]= sigma[p:2*p, p:2*p];
    res[p:2*p, 0:p]= -sigma[0:p, 0:p];
    res[p:2*p, p:2*p]= -sigma[0:p, p:2*p];
    res = res.real
    return res

### Creates a random thermal density matrix w.r.t. some Hamiltonian ###
def ThermalState(F,T):

    m = floor(F.shape[0]/2)
    V=F[0:m,0:m]; 
    Tt=F[m:2*m,m:2*m];
    w = Tt[0,0]
    V=V/w
    Tt=Tt/w
    if np.allclose(Tt, np.eye(Tt.shape[0])) == False:
        print('Error: T is not identity')
    SqrtM = sqrtm(V);
    SqrtMinv = inv(SqrtM);
    
    sigma = np.eye(2*m);
    sigma[0:m, 0:m] = SqrtMinv;
    sigma[m:2*m,m:2*m] = SqrtM 

    if T != 0:
        beta = 1/T
        arg = beta * w * SqrtM / 2
        if arg[0,0] > 7:
            coth = np.eye(m)
            coth += 2*expm(-2*arg)
            coth += 2*expm(-4*arg)
        else:
            coth = inv(tanhm(arg))
        Coth = np.eye(2*m)
        Coth[0:m, 0:m] = coth
        Coth[m:2*m,m:2*m] = coth
        sigma = sigma @ Coth
    return sigma

def ComputeHellinger(dmu,sig1,sig2,N_tom):
    Log1 = 0
    Log2 = 0    
    for k in range(dmu.shape[0]):
        Log1 += (1/8)*dmu[k]**2/((sig1[k]+sig2[k])/2)
        Log2 += (1/4)*np.log(sig1[k])
        Log2 += (1/4)*np.log(sig2[k])
        Log2 += -(1/2)*np.log(((sig1[k]+sig2[k])/2))
                
    Hellinger = np.sqrt(1-np.exp(Log2 - Log1*N_tom))
    N_half = (np.log(Log2 -np.log(1/2)) - np.log(Log1))/np.log(10)

    return Hellinger, N_half

def Tomography(a,N_tom,Med):
    if N_tom == 'Infinity':
        return a    
    N_tom = float(N_tom)
    
    r = []
    if N_tom <= 10:
        for t in range(len(a)):
            a_tom = (Med[t]+a[t])*np.random.chisquare(N_tom-1)/(N_tom -1) - Med[t]
            r.append(a_tom)
    else:
        for t in range(len(a)):
            a_tom = a[t]+(a[t]+Med[t])*np.random.randn()*np.sqrt(2/(N_tom-1))
            r.append(a_tom)
    return r

def RTemp(TMean,TDev,TDist='Uniform'):
    if TDev == 0:
        return TMean
    
    if TDist == 'Gaussian' or TDist == 'Normal':
        T = np.random.normal(TMean,TDev)
        T = max(T,0)
    if TDist == 'Uniform':
        z = 2*np.random.ranf()-1
        T = TMean + z*TDev
        T = max(T,0)
    return T
    
def PCA(X=np.array([])):
    print('Performing PCA on Data')
    (n, d) = X.shape                                  # Determine data shape

    print('Computing Mean and Covariance Matrix')
    Xm = np.mean(X, 0).real                           # Compute the mean of each colum
    X0  = X - np.tile(Xm, (n, 1))                     # Subtract off mean from each row
    Cov = np.dot(X0.T, X0)/(n-1)
    print('Done!')

    print('Computing Eigensystem')
    (lam, M) = np.linalg.eig(Cov)  # Compute eigensystem
    print('Done!')
    
    # Taking real parts and sorting
    lam = lam.real               
    M = M.real    
    idx = lam.argsort()[::-1]       
    lam = lam[idx]
    M = M[:,idx]
    M=M.T
    
    for k in range(len(lam)):
        if k == 0:
            continue
        if lam[k] < 0:
            lam[k] = lam[k-1]/10
            
    return Xm, lam, M

def PCA_Compress(lam,VarKeep):
    sum_lam = sum(lam)
    cumsum = 0
    for dim_c in range(len(lam)):
        cumsum += lam[dim_c]
        if cumsum/sum_lam > VarKeep:
            break
    dim_c += 1
    return dim_c