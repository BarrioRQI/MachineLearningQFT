#!/usr/bin/env python
# coding: utf-8

# In[7]:


import csv
import numpy as np
#import qutip as qt 
import scipy as sp
from math import floor
from numpy.linalg import inv
from scipy.linalg import expm, sqrtm, tanhm
from scipy.linalg import logm
from scipy.sparse import csr_matrix

###############################################################################
############################# DEFINITIONS #####################################
###############################################################################

#Order of appearance in datagenerator

### All the 2x2 Pauli matrices in a Qobj valued list ### - To generate probe local Hamiltonian
def SS(r,Qobj = False):
    
    if Qobj == False:
        Sigma = [np.array([[1,0],[0,1]]),
                 np.array([[0,1],[1,0]]),
                 np.array([[0,-1j],[1j,0]]),
                 np.array([[1,0],[0,-1]])]
        return Sigma[r]
    
    Sigma = [qt.identity(2),qt.sigmax(),qt.sigmay(),qt.sigmaz(), ]
    
    return Sigma[r]

def InitializeProbeState(State='Ground',Qobj=False,QGaussian=False):
    if QGaussian == False:
        if State == 'Ground':                                          # Initialize probe to ground state 
            return sum([1,0,0,-1][j]*SS(j,Qobj=Qobj)/2 for j in range(4))
        elif State == 'Plus':                                            # Initialize probe to plus state
            return sum([1,1,0,0][j]*SS(j,Qobj=Qobj)/2 for j in range(4))
        elif State == 'Excited':                                         # Initialize probe to plus state
            return sum([1,0,0,1][j]*SS(j,Qobj=Qobj)/2 for j in range(4))
        else:
            print("Error: Pick Valid Inital State")
    
    if QGaussian == True:
        if State == 'Ground':                                   # Initialize probe to ground state - Gaussian
            return np.array([[1,0],[0,1]])
        else:
            print("Error: Pick Valid Inital State")
    return np.nan()

############################################################################
##################### DEFINING HAMILTONIANS ################################
############################################################################

def ComputeHams(W,Geometry,sigma,B,D,QGaussian=False,QScalar=True):
    # Extract Energy Scales
    wS = W[0]       # Probe free energy scale
    wE = W[1]       # Environment free energy scale
    gEE = W[2]      # Environment internal coupling energy scale
    gSA = W[3]      # Probe- Environment coupling energy scale

    # Extract Environment Geometry
    LatLen = Geometry[0]    # Length of Environment Lattice
    LatDim = Geometry[1]    # Dimensions of Environment Lattice
    N_Env = LatLen**LatDim  # Number of Environment elements

    # Define Coupling Types
    if QGaussian == False:
        JEE = JCase('XX') 
        JSA = JCase('XX')        
    if QGaussian == True:
        JEE  = JCase('xx')
        JSA  = JCase('xx')

    FieldDim = 1
    if QScalar == False:
        FieldDim = LatDim

    #print('Defining Probes Local Hamiltonian')
    if QGaussian == True:   
        HS = wS * SS(0)     #In gaussian we define F with H = 1/2 R F R^T
    if QGaussian == False:
        HS = wS * SS(3)/2

    #print('Defining Environment Local Hamiltonian')
    HE0 = EnvFreeHam(N_Env,wE,gEE,FieldDim, QGaussian=QGaussian)
        
    #print('Defining Environment Internal Hamiltonian for Bulk')
    AdjBulk   = SquareLatticeAdjList(LatLen,LatDim,CutFilm=True)
    HEintBulk = -gEE * EnvIntHam(JEE,FieldDim,AdjBulk, QGaussian=QGaussian)
    
    # print('Defining Environment Internal Hamiltonian for Boundary')
    AdjBound = SquareLatticeAdjList(LatLen,LatDim,JustFilm=True)        
    if B < 4:
        HEB_thermal = -gEE * HEBound(B,FieldDim,AdjBound,QGaussian=QGaussian)
        HEB_dynamic = -gEE * HEBound(B,FieldDim,AdjBound,QGaussian=QGaussian)
    if B == 4:
        HEB_thermal = -gEE * HEBound(1,FieldDim,AdjBound,QGaussian=QGaussian)
        HEB_dynamic = -gEE * HEBound(0,FieldDim,AdjBound,QGaussian=QGaussian)

    HE_thermal = HE0+HEintBulk+HEB_thermal
    HE_dynamic = HE0+HEintBulk+HEB_dynamic
    
    #print('Defining Probe - Environment Hamiltonian')
    HSA  = -gSA * SetupSAHam(JSA,LatDim,1+D,sigma,N_Env, QGaussian = QGaussian)       # Define the system ancilla coupling
    if D > LatLen - 2: return 'Error: D too large'
    
    Ham = TotalHam(HS,HSA,HE_dynamic, LatDim, QGaussian = QGaussian)

    return Ham, HE_thermal

### Defines common coupling cases ###
def JCase(Coupling_Case):
    if Coupling_Case == 'XX':   # H12 = Sx * Sx
        J4=np.array(
         [[0,0,0,0],
          [0,1,0,0],
          [0,0,0,0],
          [0,0,0,0]])
    if Coupling_Case == 'ZZ':     # H12 = Sz * Sz
        J4=np.array(
         [[0,0,0,0],
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,1]])
    if Coupling_Case == 'SS':     # H12 = Swap
        J4=np.array(
         [[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1]])/2
    if Coupling_Case == 'Random':  # Random Coupling
        Ham = RandomHam_GUE(2)  # Pick a random 2x2 Hamiltonian
        J4 = np.zeros((4,4))    # Initialize Pauli basis representation
        for i in range(4):      # Convert Hamiltonian to Pauli basis representation
            for j in range(4):
                if i == 0 or j == 0:          # Leave off local Hamiltonians
                    J4[i,j] == 0
                else:
                    J4[i,j]=np.trace(np.kron(SS(i),SS(j)).dot(Ham)).real
    if Coupling_Case == 'xx':     # H12 = x_1 x_2
        J4=np.array([1,0])
                     
    if Coupling_Case == 'pp':     # H12 = p_1 p_2
        J4=np.array([0,1])
        
    if Coupling_Case == 'xx + pp':     # H12 = x_1 x_2 + p_1 p_2
        J4=np.array([1,1])
        
    return J4

### Generates the Environment's Free Hamiltonian                ###
### ex) for N_E=3 we want HE0 ~ Sz*1*1 + 1*Sz*1 + 1*1*Sz        ###
def EnvFreeHam(N,w,g,FieldDim, QGaussian = False):
    if QGaussian == False:
        dimE = 2**N                          # Dimension of environment Hilbert space
        Hdiag = np.zeros(dimE)               # Initializing list of eigenvalues

        for i in range(dimE):                # Set each eigenvalue 
            binarydigits = [int(d) for d in str(bin(dimE-i-1))[2:]]
            Hdiag[i] = sum(binarydigits)     

        Hdiag += -sum(Hdiag)/dimE            # Remove the constant part
        HE0 = w*np.diag(Hdiag)                 # Flesh it out
        
    if QGaussian == True:
        m=FieldDim*N
        T = w*np.eye(m)
        V = (w+2*g)*np.eye(m)
        HE0 = np.zeros((2*m,2*m))
        HE0[0:m,0:m] = V
        HE0[m:2*m,m:2*m] = T
    return HE0

### Creates the adjacency list for a d-dimensional square lattice of size L  ###
def SquareLatticeAdjList(L,d,IncludeBulk=True, IncludePeriodic = False,JustFilm=False,CutFilm=False):      
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
def EnvIntHam(J,FieldDim,adjList, QGaussian=False):    
    N_E = len(adjList)                      # Identify number of spins in environment
    d = FieldDim 
    if QGaussian == True: 
        M= np.zeros((2*d*N_E ,2*d*N_E));        
        for n_i in range(len(adjList)):                 # For each site i 
            for n_j in adjList[n_i]:                    # For each neighboring site j
                M[n_i*d :(n_i+1)*d ,n_j*d:(n_j+1)*d] = J[0]*np.eye(d);                                 # Updates xx part of the matrix
                M[n_i*d + d*N_E :(n_i+1)*d+ d*N_E, n_j*d + d*N_E:(n_j+1)*d + d*N_E] = J[1]*np.eye(d);   # Updates pp part of the matrix
        
        return M        
    
    if N_E == 1:                            # If N = 1 return 0 Hamiltonian
        return 0    

    dimE = 2**N_E                           # Initialize the coupling Hamiltonian
    Hc = np.zeros((dimE,dimE),dtype=complex)
    for n_i in range(len(adjList)):                 # For each site i 
        for n_j in adjList[n_i]:                    # For each neighboring site j
            if n_i < n_j:
                for r in range(4):                      # For each pauli on site n_i
                    for t in range(4):                  # For each pauli on site n_j
                        Hc += J[r,t]*SS_xyx_ijk_N([r,t],[n_i,n_j],N_E)    
    return Hc

def HEBound(B,FieldDim,AdjBound,QGaussian=False):
    if QGaussian == False:
        if B == 0:
            HEint = 1 * EnvIntHam(JCase('XX'),FieldDim,AdjBound,QGaussian = QGaussian)           # Hij =  gEE Xi Xj
        elif B == 1:
            HEint = 0 * EnvIntHam(JCase('XX'),FieldDim,AdjBound,QGaussian = QGaussian)           # Hij =  0
        elif B == 2:
            HEint = (1/2) * EnvIntHam(JCase('XX'),FieldDim,AdjBound,QGaussian = QGaussian)       # Hij =  gEE Xi Xj / 2
        elif B == 3:
            HEint = 1 * EnvIntHam(JCase('ZZ'),FieldDim,AdjBound,QGaussian = QGaussian)           # Hij =  gEE Zi Zj
        else:
            print('Error: Pick valid boundary case')
    if QGaussian == True:
        if B == 0:
            HEint = 1 * EnvIntHam(JCase('xx'),FieldDim,AdjBound,QGaussian = QGaussian)
        elif B == 1:
            HEint = 0 * EnvIntHam(JCase('xx'),FieldDim,AdjBound,QGaussian = QGaussian)
        elif B == 2:
            HEint = (1/2) * EnvIntHam(JCase('xx'),FieldDim,AdjBound,QGaussian = QGaussian)
        elif B == 3:
            HEint = 1 * EnvIntHam(JCase('pp'),FieldDim,AdjBound,QGaussian = QGaussian)
        else:
            print('Error: Pick valid boundary case')
    return HEint

def SetupSAHam(JSA,FieldDim,n_A,sigma,N_E, QGaussian = False):    
    d = FieldDim
    if QGaussian == True: 
        F= np.zeros((2*d*N_E +2*d,2*d*N_E +2*d));
        for x in range(N_E):
            weight = np.exp(-(x-n_A)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
            F[d*N_E:d*N_E+d, x:x+d] = weight*JSA[0]*np.eye(d);            # Updates xx part of the matrix
            F[x:x+d,d*N_E:d*N_E+d]  = weight*JSA[0]*np.eye(d);
        
            F[2*d*N_E +d:2*d*N_E +2*d, x+d*N_E:x+d+d*N_E] = weight*JSA[1]*np.eye(d);  # Updates pp part of the matrix
            F[x+d*N_E:x+d+d*N_E,2*d*N_E +d:2*d*N_E +2*d]  = weight*JSA[1]*np.eye(d);
    
        return F       
                
    dim = 2**(1+N_E)                            # Define the Hilbert space dimension
    HSA = np.zeros((dim,dim), dtype=complex)    # Initialize the Hamiltonian

    for x in range(4):                          # Couple the probe to site n_A via JSA
        for y in range(4):
            HSA += JSA[x,y] * SS_xyx_ijk_N([x,y],[0,1+n_A],N_E+1)

    return HSA

### Picks which coupling case and constructs the total Hamiltonian ###
def TotalHam(HS,HSA,HE,d, QGaussian = False):
    
    if QGaussian == True:
        HS1 = directsum(np.zeros(HE.shape),HS)
        HE1 = directsum(HE, np.zeros(HS.shape))
        F = HSA + HS1 + HE1
        return F

    N_S = int(np.log2(HS.shape[0]))           # Determine number of spins in probe
    N_E = int(np.log2(HE.shape[0]))           # Determine number of spins in environment
    HS_otimed = np.kron(HS,Id(N_E))           # Tensor HS to full dimensions
    HE_otimed = np.kron(Id(N_S),HE)           # Tensor HE to full dimensions 
    Ham = HS_otimed + HE_otimed + HSA         # Add up total Hamiltonian        

    return Ham

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
def DefProjList(Ham,N_t,Tmin,Tmax, QGaussian=False):
    dim = Ham.shape[0]
    if Tmin == 0: Tmin = Tmax/N_t
    Tlist=np.linspace(Tmin,Tmax,N_t)              # Divide evolution period into k pieces
    dT = Tlist[1]-Tlist[0]                        # Compute time step
    
    if QGaussian == False:
        Ulist   = np.zeros((N_t,dim,dim),dtype=complex) # Initialize a list of unitaries
        UAdlist = np.zeros((N_t,dim,dim),dtype=complex) # Initialize a list of unitaries
        Ustep   = expm(-1.0j * Ham * dT)                # Compute time step unitary
        Ulist[0] = expm(-1.0j * Ham * Tmin)           # Compute initial unitary    
        UAdlist[0] = np.transpose(np.conj(Ulist[0]))
        for k in range(1,N_t):
            Ulist[k] = Ustep @ Ulist[k-1]
            UAdlist[k] = np.transpose(np.conj(Ulist[k]))

    if QGaussian == True:
        Ulist = np.zeros((N_t,dim,dim)) # Initialize a list of unitaries
        UAdlist = np.zeros((N_t,dim,dim)) # Initialize a list of unitaries
        Ustep = expm(np.asarray(multOmega(Ham)*dT))
        Ulist[0] = expm(np.asarray(multOmega(Ham)*Tmin))
#        Ustep = Sexp(Ham,dT)
#        Ulist[0] = Sexp(Ham,Tmin)           # Compute initial unitary
        UAdlist[0] = np.transpose(Ulist[0])
        for k in range(1,N_t):
            Ulist[k] = Ustep @ Ulist[k-1]
            UAdlist[k] = np.transpose(Ulist[k])        

    if QGaussian == False:
        N_Env = int(np.log2(dim))
        Proj0 = [sp.sparse.kron(SS(1+r),Id(N_Env,Sparse=True),format="csr") for r in range(3)]
        ProjList = np.zeros((N_t,3,2*dim,2*dim), dtype=complex) 

    if QGaussian == True:
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
    res =np.matrix([[0 + 0j  for i in range(0,2*p)] for j in range(0,2*p)])
    res[0:p, 0:p]= sigma[p:2*p, 0:p];
    res[0:p, p:2*p]= sigma[p:2*p, p:2*p];
    res[p:2*p, 0:p]= -sigma[0:p, 0:p];
    res[p:2*p, p:2*p]= -sigma[0:p, p:2*p];
    return res

### Returns the N qubit identity matrix with appropriate tensor product structure ###
def Id(N,Qobj = False,Sparse = False):
    if Qobj == False:
        IdN = np.diag([1]*(2**N))
        if Sparse == True:
            IdN = csr_matrix(IdN)           
        return IdN

    dim = 2**N                           # dimension of Hilbert space
    dims = [[2]*N,[2]*N]                 # QuTip tensor structure 
    values = np.diag([1]*dim)            # Values of entries 
    IdN = qt.Qobj(values, dims = dims)   # Put it all together

    return IdN

### Creates a random thermal density matrix w.r.t. some Hamiltonian ###
def ThermalState(Ham,T, QGaussian=False):
    
    if QGaussian==True:
        return GaussThermal(Ham, T)        
        
    try:
        N_E = len(Ham.dims[0])
        Qobj = True
    except AttributeError:
        N_E = int(np.log2(Ham.shape[0]))
        Ham = qt.Qobj(Ham)
        Qobj = False
    
    try:
        Beta = 1/T
    except ZeroDivisionError:
        Beta = 10**3/Ham.norm()
    
    rho = (- Beta * Ham).expm()      # Exponentiate the scaled Hamiltonian
    rho = rho/rho.tr()                # Normalize the state
    
    if Qobj == False:
        rho = rho.full()    
    return rho


def GaussThermal(F, T):
        
    m = floor(F.shape[0]/2)
    V=F[0:m,0:m]; 
    Tt=F[m:2*m,m:2*m];
    w = Tt[0,0]
    
    S = sqrtm(Tt.dot(inv(V)));
    Sinv = inv(S);
    
    if T==0:
        sigma =np.eye(2*m);
        sigma[0:m, 0:m]= S;
        sigma[m:2*m,m:2*m]=Sinv

    else:
        beta = 1/T;
        cot=inv(tanhm(beta*w*S));
        sigma =np.eye(2*m);
        sigma[0:m, 0:m]= S.dot(cot);
        sigma[m:2*m, m:2*m]=Sinv.dot(cot)
    
    
    return sigma

def partialsigmaD(sigma):
    m = floor(sigma.shape[0]/2);    
    
    B= np.matrix([[0 + 0j  for i in range(0,2)] for j in range(0,2)]);
    B[0,0] = sigma[m-1,m-1];
    B[0,1] = sigma[m-1,2*m-1];
    B[1,0] = sigma[2*m-1,m-1];
    B[1,1] =sigma[2*m-1,2*m-1]; 
    
    return B

Sig = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])

def Tomography(a,N_tom,Med, QGaussian = False):
    if N_tom == 'Infinity':
        return a    
    N_tom = float(N_tom)
    
    if QGaussian == True: 
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

    r=[]
    for n in range(len(a)):
        if a[n]+Med[n] <= -1:                 # if a <= -1
            a_tom[n] = -1 - Med[n]            # we always measured -1
        elif a[n] >= 1:                       # if a >= 1
            a_tom[n] = 1 - Med[n]             # we always measured +1
        else:                                 # otherwise
            p = (1+Med[n]+a[n])/2                    # we measure +1 with probability p
            if N_tom <= 10:
                a_tom[n] = 2*np.random.binomial(N_tom,p)/N_tom - 1 -Med[n]
            else:
                a_tom[n] = a + 2*np.sqrt(p*(1-p)/N_tom)*np.random.randn() # finding +1 in B of N_tom measurements
    return a_tom


































### Builds S_{x,i} * S_{y,j} * S_{z,k} padded with I_2's between, before and after to length N_E ###
def SS_xyx_ijk_N(paulis,sites,N, Qobj=False):

    if 0 in sites:
        pos = sites.index(0)
        SSS = SS(paulis[pos],Qobj=Qobj)
    else:
        SSS = SS(0,Qobj=Qobj)
            
    for k in range(1,N):
        if k in sites:
            pos = sites.index(k)
            if Qobj == False:
                SSS = np.kron(SSS,SS(paulis[pos],Qobj = False))
            else:
                SSS = qt.tensor(SSS,SS(paulis[pos],Qobj = True))                
        else:
            if Qobj == False:
                SSS = np.kron(SSS,SS(0,Qobj = False))
            else:
                SSS = qt.tensor(SSS,SS(0,Qobj = True))                

    return SSS



### Finds the neighbors of the neighbors... (x k) of site zero ###
def Neighbors(k,adjList):      
    AL=[0]                              # 0's only 0th neighbor is itself
    ALn=AL                              # ALn is our working list
    for n in range(1,k+1):              # n is just a counter
        for ind in AL:                  # for all neighbor from the previous step
            ALn = ALn + adjList[ind]    # look up and add their neighbors
            ALn = list(set(ALn))        # remove duplicates from the list
        AL = ALn                        # finalize the list and start over
    return AL

### Returns a Random Hamiltonian for N qubits from the Gaussian Unitary Ensemble ###
def RandomHam_GUE(N,Qobj=False):
    dim = 2**N                                        # dimension of Hilbert space
    dims = [[2]*N,[2]*N]                              # QuTip tensor structure 
    Hreal = np.random.normal(0.0,1.0,(dim,dim))       # Pick dim random vectors in R^dim
    Himag = np.random.normal(0.0,1.0,(dim,dim))       # Pick dim random vectors in R^dim
    H = Hreal + 1j*Himag                              # Create dim random vectors in C^dim
    Ham = (np.transpose(np.conj(H))+H)/(2*np.sqrt(dim)) # Make it Hermitian    
    if Qobj == True:
        Ham = qt.Qobj(H, dims = dims)                     # Create the Qobj
    return Ham





### Picks a random Hamiltonian built from random 1 site modifications ###
def RandomHamiltonian1site(N_E,Qobj=False):
    dimE = 2**N_E
    dims = [[2]*N_E,[2]*N_E]
    
    ### H1 in span(1*1*...*1*SS*1*...*1)
    H1 = np.zeros((dimE,dimE), dtype=complex)
    if Qobj == True:
        H1 = qt.Qobj(H1, dims = dims)

    for i in range(N_E):        # for each site in the environment
        RH = RandomHam_GUE(1,Qobj=Qobj)   # pick a random 1 qubit Hamiltonian
        if 0 == i:              # Start building Hi
            Hi = RH             # with RH
        else:
            Hi = SS(0,Qobj=Qobj)          # or with identity
        for k in range(1,N_E):  # continue building Hi
            if k == i:
                if Qobj == True:
                    Hi = qt.tensor(Hi,RH)   # with RH
                else:
                    Hi = np.kron(Hi,RH)
            else:
                if Qobj == True:
                    Hi = qt.tensor(Hi,SS(0,Qobj = True)) # or identity
                else:
                    Hi = np.kron(Hi,SS(0,Qobj = False))
        H1 += Hi                # Add Hi to the total
    return H1

### Picks a Hamiltonian built from random 2 site modification for each pair of sites ###
def RandomHamiltonian2site(N_E,Qobj=False):

    dimE = 2**N_E
    dims = [[2]*N_E,[2]*N_E]

    ### H2 in span(1*...*1*SS*1*...*1*SS*1*...*1) ###  
    H2 = np.zeros((dimE,dimE), dtype=complex)
    if Qobj == True:
        H2 = qt.Qobj(H2, dims = dims)
    
    for n_i in range(N_E):                 # For each site i
        for n_j in range(N_E):             # For each site j
            if n_i < n_j:
                Ham = RandomHam_GUE(2,Qobj=Qobj)     # Pick a random 2 qubit Hamiltonian
                J4 = np.zeros((4,4))       # Initialize Pauli basis representation
                for x in range(4):         # Convert Hamiltonian to Pauli basis representation
                    for y in range(4):
                        if Qobj == True:
                            J4[x,y]=(qt.tensor(SS(x,Qobj=True),SS(y,Qobj=True)) * Ham).tr().real
                        else:
                            J4[x,y]=np.trace(np.kron(SS(x),SS(y)).dot(Ham)).real
                for x in range(4):         # Add up weighted paulis
                    for y in range(4):
                        H2 += J4[x,y]*SS_xyx_ijk_N([x,y],[n_i,n_j],N_E,Qobj=Qobj)
    return H2



def RTemp(TMean,TDev,TDist):
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
    
def FireWall(Rho):
    try:
        N = len(Rho.dims[0])
        dims = Rho.dims
        Qobj = True
    except AttributeError:
        N = int(np.log2(Rho.shape[0]))
        dims = [[2]*N,[2]*N]
        Rho = qt.Qobj(Rho,dims=dims)
        Qobj = False    
                
    FireRho = Rho.ptrace(0)
    for k in range(1,N):
        ReducedRho = Rho.ptrace(k)
        FireRho = qt.tensor(FireRho,ReducedRho)
    
    if Qobj == False:
        FireRho = FireRho.full()

    return FireRho

### Work out what dynamics would be in this case ###
'''def FireWallAtDistJ(Rho,adjList,J):
    N_E = len(Rho.dims[0])
    inlist = Neighbors(J,adjList)
    outlist = list(set(range(N_E))-set(inlist))
    RhoIn = Rho.ptrace(inlist)
    RhoOut = Rho.ptrace(outlist)    
    FireRho = qt.tensor(RhoIn,RhoOut) 
    FireRho = FireRho.permute(inlist+outlist)
    return FireRho
    '''

def PickEnvStateTherm(y,HE0,Hint_p,Hint_n,TDist,TMean,TDev,NTDev):

    try:
        N_E = len(HE0.dims[0])
        Qobj = True
    except AttributeError:
        N_E = int(np.log2(HE0.shape[0]))
        Qobj = False
    
    dimE = 2**N_E
    dims = [[2]*N_E,[2]*N_E]
    
    T = RTemp(TMean,TDev,TDist)
    rhoth = ThermalState(HE0+Hint_p,T)
    
    if y == 0:                              # Thermal
        rhoA = rhoth

    if y == 1:                              # Too Hot
        T = RTemp(TMean+2*TDev,TDev,TDist)
        rhoA = ThermalState(HE0+Hint_p,T)

    if y == 2:                              # Too Cold
        T = RTemp(TMean-2*TDev,TDev,TDist)
        rhoA = ThermalState(HE0+Hint_p,T)
    
    
    return T, rhoA






### Picks a random initial state for the environment ###
def PickEnvState(y,HE0,Hint_p,Hint_n,TDist,TMean,TDev,NTDev):

    try:
        N_E = len(HE0.dims[0])
        Qobj = True
    except AttributeError:
        N_E = int(np.log2(HE0.shape[0]))
        Qobj = False
    
    dimE = 2**N_E
    dims = [[2]*N_E,[2]*N_E]
    
    T = RTemp(TMean,TDev,TDist)
    rhoth = ThermalState(HE0+Hint_p,T)
    
    if y == 0:                              # Thermal
        rhoA = rhoth

    if y == 1:                              # Too Hot
        T = RTemp(TMean+2*TDev,TDev,TDist)
        rhoA = ThermalState(HE0+Hint_p,T)

    if y == 2:                              # Too Cold
        T = RTemp(TMean-2*TDev,TDev,TDist)
        rhoA = ThermalState(HE0+Hint_p,T)
    
    if y == 3:                              # Thermal state w.r.t. HE plus 1 site nudges
        dH1 = RandomHamiltonian1site(N_E,Qobj=Qobj)
        dH1 = NTDev*dH1
        rhoA = ThermalState(HE0+Hint_p+dH1,T)
    
    if y == 4:                              # Thermal state w.r.t. HE plus 2 site nudges
        dH2 = RandomHamiltonian2site(N_E,Qobj=Qobj)
        dH2 = NTDev*dH2/N_E
        rhoA = ThermalState(HE0+Hint_p+dH2,T)
    
    if y == 5:                              # Thermal state w.r.t. HE plus nonlocal nudges
        dH3 = RandomHam_GUE(N_E,Qobj=Qobj)
        dH3 = N_E*NTDev*dH3
        rhoA = ThermalState(HE0+Hint_p+dH3,T)

    if y == 6:                              # Thermal state w.r.t. HE0
        rhoA = ThermalState(HE0,T)

    if y == 7:                              # Fire Wall'd
        rhoA = FireWall(rhoth)
    
    if y == 8:                              # Ginibre random density matrix
        T = 0
        rhoA = qt.rand_dm_ginibre(dimE,rank=None,dims = dims)
        if Qobj == False:
            rhoA = rhoA.full()
    
    if y == 9:                              # Haar random pure state
        T = 0
        rhoA = qt.ket2dm(qt.rand_ket_haar(dimE,dims = [[2]*N_E,[1]*N_E]))
        if Qobj == False:
            rhoA = rhoA.full()

    if y >= 10:                              # non-periodic boundary condition
        rhoA = ThermalState(HE0+Hint_n,T)
        
#    SD = StatDist(rhoth,rhoA)

    return T, rhoA

def StatDist(rho,sigma):
    try:
        rho = rho.full()
        sigma = sigma.full()
    except AttributeError:
        pass

    logr = logm(rho)
    try:
        logs = logm(sigma)
    except ValueError:
        logs = logm((sigma+rho)/2)          #To Do: This case should be triggering but isn't
        print("Sigma was singlular.")
    RE = np.trace(rho.dot(logr-logs)).real
    SD = np.sqrt(2*RE)
    return SD









def LoadDataFrom(InputFile,N_s):
    print("Beginning to Load Data from...",InputFile)
    data=[]
    with open(InputFile, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for r, row in enumerate(reader):
            if r+1 > N_s:
                break
            try:
                frow = [float(k) for k in row]
            except ValueError:
                print("Skipping row",r+1,": bad string")
                continue
            if r == 0:
                firstlen=len(row)
            newlen=len(row)
            if firstlen != newlen:
                print("Skipping row",r+1,": wrong length")
                continue
            data.append(frow)
            if (r+1) % 1000 == 0:
                 print("Loaded: ",int((r+1)/1000)," k datapoints")
    csvFile.close()
    data=np.array(data) # Work on speeding this up

    indices_shuffled = np.random.permutation(data.shape[0])    # Pick random permutation of data
    data = data[indices_shuffled,:]
    
    print("Done Loading Data!")
    return data

def ParseLabels(data,CaseLabels,LeaveOff):
    LO=int(LeaveOff)
    X = data[:,:-LO]            # X is all our data points 
    y=data[:,-1:]               # y labels thermal state, y=0, and nonthermal states y=1
    y=y.flatten()

    ylist = []
    for s in range(len(y)):
        if y[s] in ylist:
            pass
        else:
            ylist.append(int(y[s]))
    
    for s in range(len(y)):
        if y[s] in ylist:
            y[s] = ylist.index(y[s])
    
    NewCaseLabels=[1]*len(ylist)
    for r in range(len(ylist)):
        NewCaseLabels[r] = CaseLabels[ylist[r]]
    
    return X, y, NewCaseLabels

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

def LogProb(Atom,A,Ntom,CentralLimit=False):
    logptot = 0    

    if CentralLimit == True:
        for n in range(A.shape[0]):
            sigma = np.sqrt(max(0,(1-A[n]**2)/Ntom))
            if sigma != 0:
                z = (Atom[n]-A[n])/np.sqrt(sigma)
                p = np.exp(-z**2/2)/np.sqrt(2*np.pi*sigma**2)
                logptot += np.log(p)
            elif Atom[n] != A[n]:
                logptot = -np.inf
                break
            else:
                continue
        return logptot
    
    for r in range(A.shape[0]):
        pbin = (1+A[r])/2
        ntom = int(np.round(Ntom*(1+Atom[r])/2))
        if 0 < pbin < 1:                
            p = sp.special.binom(Ntom, ntom) * pbin**ntom * (1-pbin)**(Ntom-ntom)
            logptot += np.log(p)
        elif pbin < 0:
            if ntom != 0:
                logptot = - np.inf
                break
        else:
            if ntom != Ntom:
                logptot = - np.inf
                break
    return logptot

def RelLogLike(Atom,A,Ntom,Offset,QGaussian=False):
    z = - Offset
    if QGaussian == False:
        z = z/Ntom
        for k in range(A.shape[0]):        
            ptrue = (1+A[k])/2
            ptomo = (1+Atom[k])/2
            if 0 < ptrue < 1:                
                z += ptomo*np.log(ptrue) + (1-ptomo)*np.log(1-ptrue)
            elif ptrue < 0:
                if ptomo != 0:
                    z = - np.inf
                    break
            else:
                if ptomo != 1:
                    z = - np.inf
                    break
        z = Ntom * z

    if QGaussian == True:
        n = Ntom-1
        z = z/n
        for k in range(A.shape[0]):        
            r = Atom[k]/A[k]
            z += (1/2-1/n)*np.log(r)-(r-1)/2
        z = n * z

    return z


# In[ ]:




