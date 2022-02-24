#! /usr/bin/python
import numpy as np
from math import floor
from numpy.linalg import inv, eig
from numpy.linalg.linalg import det
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
    HS = wD * np.eye(2)     #In gaussian we define F with H = 1/2 R F R^T
    HE0 = np.zeros((2*N_Env,2*N_Env))
    HE0[0:N_Env,0:N_Env] = (mcc+2/mcc) * np.eye(N_Env)
    HE0[N_Env:2*N_Env,N_Env:2*N_Env] = mcc * np.eye(N_Env)
        
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
    
    Ustep = expm(np.asarray(multOmega(Ham)*dT))
    Ulist[0] = expm(np.asarray(multOmega(Ham)*Tmin))

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

        if arg[0,0] > 10:
            #print('greg')
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

def Tomography(a, N_tom, Med):
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
    (n, d) = X.shape                                  # Determine data shape

    Xm = np.mean(X, 0).real                           # Compute the mean of each colum
    X0  = X - np.tile(Xm, (n, 1))                     # Subtract off mean from each row
    Cov = np.dot(X0.T, X0)/(n-1)

    (lam, M) = np.linalg.eig(Cov)  # Compute eigensystem
    
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

######################################################################################
# ------------------------ Fock and PAC state sensing utils ------------------------ #

fock_moments = np.array([
    [1, 1/2, 3  /4,   15/8,   105/16], 
    [1, 3/2, 15 /4,  105/8,   945/16], 
    [1, 5/2, 39 /4,  375/8,  4305/16], 
    [1, 7/2, 75 /4,  945/8, 13545/16], 
    [1, 9/2, 123/4, 1935/8, 33705/16]
])

pac_moments = np.array([
    [1, 1/2, 3  /4,   15/8,   105/16], 
    [1, 3/2, 21 /4,  215/8,  2835/16], 
    [1, 5/2, 51 /4,  715/8, 12425/16], 
    [1, 7/2, 93 /4, 1635/8, 34755/16], 
    [1, 9/2, 147/4, 3095/8, 77385/16]
])

def get_fock_moment(n, m):
    return fock_moments[n, m]

def get_pac_moment(n, m):
    return pac_moments[n, m]

def generate_u_v(F, wD, L, t):
    l = len(F)
    m = len(F)*2 + 2 # DO NOT FIX

    u = np.zeros((m, 1))
    u[0] = np.cos(wD*t)
    u[l+1] = np.sin(wD*t)

    v = np.zeros((m, 1))
    for n in range(len(F)):
        w = (n+1)*np.pi/L
        v[n+1] = F[n]*np.cos(w*t)/np.sqrt(w)
        v[n + l + 2] = F[n]*np.sin(w*t)*np.sqrt(w)

    return u, v

def generate_S_exp(sigma, L, wD, t, lam, N_cutoff, inv=True):
    F = get_sine_gaussian_fourier_coefficients(sigma, L, N_cutoff)
    u, v = generate_u_v(F, wD, L, t)
    
    M = u*np.transpose(v) + v*np.transpose(u)
    M = multOmega(M)

    if inv:
        S = np.eye(len(M)) - lam*M
    else:
        S = np.eye(len(M)) + lam*M

    return S

def generate_two_delta_time_evolve_S(sigma, L, wD, t1, t2, lam, N_cutoff, inv=True):
    assert(t2 > t1)
    S1 = generate_S_exp(sigma, L, wD, t1, lam, N_cutoff, inv)
    S2 = generate_S_exp(sigma, L, wD, t2, lam, N_cutoff, inv)

    assert(np.isclose(det(S1), 1, 7))
    assert(np.isclose(det(S2), 1, 7))

    if inv:
        return S1 @ S2
    else:
        return S2 @ S1

from scipy.special import erf

def get_sine_gaussian_fourier_coefficients(sigma, L, N):
    # should be calculated with Mathematica, if calculated with Numpy would be an approximation
    # since we are doing a simulation, everything is exact until measurement noise is introduced
    F = np.zeros(N)
    b = np.pi*sigma/(np.sqrt(2)*L)
    a = L/(np.sqrt(8)*sigma)

    for n in range(1,N+1,2):
        F[n-1] = ((-1)**((n-1)//2))*np.exp(-(n*b)**2)*((erf(complex(a, -n*b)) + erf(complex(a, n*b))).real)/np.sqrt(2*L)

    return F

def get_cosine_gaussian_fourier_coefficients(sigma, L, N):
    # should be calculated with Mathematica, if calculated with Numpy would be an approximation
    # since we are doing a simulation, everything is exact until measurement noise is introduced

    F = np.zeros(N)
    b = np.pi*sigma/(np.sqrt(2)*L)
    a = L/(np.sqrt(8)*sigma)

    for n in range(2,N,2):
        F[n] = ((-1)**(n//2))*np.exp(-(n*b)**2)*((erf(complex(a, -n*b)) + erf(complex(a, n*b))).real)/np.sqrt(2*L)

    F[0] = erf(a)/np.sqrt(2*L)
    return F

def generate_moment(N, M, state, quadrature, t1, t2, lam, N_cutoff, sigma, L, wD):
    # N = avg number of excitations in lowest mode
    # M = moment
    # quadrature = vector picking out quadrature of interest, should be unit vector 
    # N_cutoff is number of modes of the field considered
    
    S = generate_two_delta_time_evolve_S(sigma, L, wD, t1, t2, lam, N_cutoff, inv=True)

    Q = (S.T) @ quadrature

    assert(np.any(np.isnan(Q)) == False)

    l = Q.shape[0]
    m = 0
    T2 = get_fock_moment(0, 1)*np.ones(l)
    T4 = get_fock_moment(0, 2)*np.ones(l)
    T6 = get_fock_moment(0, 3)*np.ones(l)
    T8 = get_fock_moment(0, 4)*np.ones(l)
    Q = Q.flatten()
    QP = np.zeros(Q.shape[1])
    QP[:] = Q[:]
    Q = QP

    first_mode_moments = []
    for i in range(4):
        first_mode_moments.append(get_fock_moment(N, i+1) if state == 'fock' else get_pac_moment(N, i+1))

    for i, T in enumerate([T2, T4, T6, T8]):
        T[1] = first_mode_moments[i]
        T[l//2 + 1] = first_mode_moments[i]

    Q2 = np.square(Q)
    Q4 = np.square(Q2)
    Q6 = Q4*Q2
    Q8 = np.square(Q4)

    if M == 2:
        m += np.sum(Q2 * T2)

    if M == 4:
        m += np.sum(Q4 * (T4 - 6*np.square(T2))) + 6*np.sum(Q2 * T2)**2

    if M == 8:
        #A1 = np.sum(Q8*T8)
        #A2 = np.sum(Q4*T4)**2 - np.sum(Q8*T4*T4)
        #A3 = np.sum(Q4*T4)*(np.sum(Q2*T2)**2) - np.sum(Q4*T4)*np.sum(Q4*T2*T2) \
        #    - 2*np.sum(Q6*T4*T2)*np.sum(Q2*T2) + 2*np.sum(Q8*T4*T2*T2)
        #A4 = np.sum(Q2*T2)**4 - 6*np.sum(Q4*T2*T2)*(np.sum(Q2*T2)**2) + 3*np.sum(Q4*T2*T2)**2 \
        #    + 8*np.sum(Q6*T2*T2*T2)*np.sum(Q2*T2) - 6*np.sum(Q8*T2*T2*T2*T2)
        #A5 = np.sum(Q6*T6)*np.sum(Q2*T2) - np.sum(Q8*T6*T2)

        #m += A1 + 35*A2 + 210*A3 + 105*A4 + 28*A5

        m = 0
        m += np.sum(Q8 * (T8 - 28*T6*T2 - 35*T4*T4 + 420*T4*T2*T2 - 630*T2*T2*T2*T2))
        m += np.sum(Q6 * (28*T6 - 420*T4*T2 + 840*T2*T2*T2))*np.sum(Q2 * T2)
        m += np.sum(Q4 * T4)*np.sum(Q4 * (35*T4 - 210*T2*T2 ))
        m += (np.sum(Q2 * T2)**2)*np.sum(Q4 * (210*T4 - 630*T2*T2))
        m += 105*(np.sum(Q2 * T2)**4 + 3*(np.sum(Q4*T2*T2)**2))

    assert(np.isnan(m) == False)

    return m

def get_quadrature(a, b, N_cutoff):
    # return the 
    q = np.zeros((2*N_cutoff + 2, 1))
    assert(np.isclose(a**2 + b**2, 1, 6))
    q[0] = a
    q[N_cutoff + 1] = b

    return q

def moment_Tomography(a, a_p, N_tom):
    # add tomographic noise to an even moment for large N_tom
    # a = value Nth even moment
    # a_p = value of 2Nth moment
    if N_tom == 'Infinity':
        return a    
    N_tom = float(N_tom)
    
    r = []
    for t in range(len(a)):
        a_tom = a[t]+np.sqrt((a_p[t] - (a[t])**2)/N_tom)*np.random.randn()
        r.append(a_tom)
    return r