#! /usr/bin/python
import numpy as np
from math import floor
from numpy.linalg import inv, eig
from numpy.linalg.linalg import det
from scipy.linalg import expm, sqrtm, tanhm
from scipy.special import dawsn, expi
import smearing_functions as sf
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

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

######################################################################################
# ------------------------ Step switching + 1D discretized lattice ------------------------ #

def get_symplectic_generator(wD,m,lambda_,lattice_length,sigma,B,D,dim=1,smearing='gaussian',cutoff='none',eps=None):
    N_lat = int(lattice_length)
    FD = wD*np.eye(2)
    size=2*(N_lat**dim)
    F0_uncoupled = np.zeros((size,size))
    F0_uncoupled[0:size//2,0:size//2] = (m+(2*dim/m if m != 0 else 2*dim)) * np.eye(size//2)
    F0_uncoupled[size//2:size,size//2:size] = m*np.eye(size//2)
    
    include_periodic = False 
    if B==4:
        include_periodic = True

    F0_adjacency_list = SquareLatticeAdjList(lattice_length, dim, IncludePeriodic=include_periodic)
    F0_adjacent_int = ((1/m) if m != 0 else 1)* Fint_unitless(F0_adjacency_list)
    
    if B == 1 or B == 4:
        F0_dynamic = F0_uncoupled + F0_adjacent_int
        F0_thermal = F0_uncoupled + F0_adjacent_int
    elif B == 2:
        F0_adjacent_int[0,1] = 0
        F0_adjacent_int[1,0] = 0
        F0_dynamic = F0_uncoupled + F0_adjacent_int
        F0_thermal = F0_uncoupled + F0_adjacent_int
    elif B == 3:
        F0_dynamic = F0_uncoupled + F0_adjacent_int
        F0_adjacent_int[0,1] = 0
        F0_adjacent_int[1,0] = 0
        F0_thermal = F0_uncoupled + F0_adjacent_int

    if D > lattice_length: print('Error: D too large')
    Fint  = lambda_ * get_Fint(D,sigma,N_lat,eps,wD,smearing,cutoff, dim)       # Define the system ancilla coupling
    F = get_Ftotal(F0_dynamic,FD,Fint)
    return F, F0_thermal

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
def Fint_unitless(adjList):    
    N_E = len(adjList)                      # Identify number of degrees of freedom
    F = np.zeros((2*N_E ,2*N_E));        
    for n_i in range(len(adjList)):                 # For each site i 
        for n_j in adjList[n_i]:                    # For each neighboring site j
            F[n_i,n_j] = -1;  
    return F                               # Updates xx part of the matrix    return F

def HEBound(B,AdjBound):
    if B == 1:
        HEint = 1 * Fint_unitless(AdjBound)
    elif B == 2:
        HEint = 0 * Fint_unitless(AdjBound)
    else:
        print('Error: Pick valid boundary case')
    return HEint

def get_Fint(n_A, sigma, N_lat, eps=None, wD=None, smearing='gaussian',cutoff='none', dim=1):    
    n_A = n_A - 1
    if cutoff == 'none':
        SF = sf.get_smearing_function(smearing, FT=False)
    else:
        SF = sf.get_smearing_function(smearing, FT=True)
        C = sf.get_cutoff_function_FT(cutoff)
        def F_times_C(k, sigma, eps, omega, F, C):
            return SF(k, sigma)*C(k, eps, omega)

    size = 2*(N_lat**dim) + 2
    Fint = np.zeros((size, size));
    for x in range(N_lat**dim):
        del_x = 0
        if dim == 1:
            del_r = x - n_A
        elif dim == 2:
            del_y = x//N_lat - n_A
            del_x = x%N_lat - n_A
            del_r = np.sqrt(del_x**2 + del_y**2)
        
        if cutoff == 'none':
            weight = SF(del_r, sigma)

        #elif (eps != None) and (wD != None):
        #    weight = get_IFT(F_times_C, x - n_A, (sigma, eps, wD, F, C))
        #    weight = weight.real

        Fint[N_lat**dim,x] = weight
        Fint[x,N_lat**dim]  = weight

    return Fint
### Picks which coupling case and constructs the total Hamiltonian ###
def get_Ftotal(F0,FD,Fint):
    FD1 = directsum(np.zeros(F0.shape),FD)
    F01 = directsum(F0, np.zeros(FD.shape))
    F = Fint + FD1 + F01

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
def getProjectorList(F,N,tmin,tmax):
    # F = symmetric matrix for symplectic evolution
    # N = number of measurements per window

    dim = F.shape[0]
    if tmin == 0: tmin = tmax/N
    tlist=np.linspace(tmin,tmax,N)              # Divide evolution period into k pieces
    dt = tlist[1]-tlist[0]                        # Compute time step
    
    Ulist = np.zeros((N,dim,dim)) # Initialize a list of unitaries
    Udaggerlist = np.zeros((N,dim,dim)) # Initialize a list of unitaries

    Ustep = expm(np.asarray(multOmega(F)*dt))
    Ulist[0] = expm(np.asarray(multOmega(F)*tmin))

    Udaggerlist[0] = np.transpose(Ulist[0])
    for k in range(1,N):
        Ulist[k] = Ustep @ Ulist[k-1]
        Udaggerlist[k] = np.transpose(Ulist[k])

    # isolate pD, qD, (pD + qD)/sqrt{2}
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
    ProjList = np.zeros((N,3,2*m,2*m))

    for n in range(N):
        for r in range(3):
            ProjList[n,r] = Udaggerlist[n] @ Proj0[r] @ Ulist[n]

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
    SqrtM = np.real(sqrtm(V));
    SqrtMinv = inv(SqrtM);
    
    sigma = np.eye(2*m);
    sigma[0:m, 0:m] = SqrtMinv;
    sigma[m:2*m,m:2*m] = SqrtM 

    if T != 0:
        beta = 1/T
        arg = beta * w * SqrtM / 2

        if arg[0,0] > 10:
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

def InitializeProbeState(State='Ground'):
    if State == 'Ground':                                   # Initialize probe to ground state - Gaussian
        return np.array([[1,0],[0,1]])
    else:
        print("Error: Pick Valid Inital State")
    return np.nan()

def Tomography(a, N_tom, med):
    if N_tom == 'Infinity':
        return a    
    N_tom = float(N_tom)
    
    r = []
    if N_tom <= 10:
        for t in range(len(a)):
            a_tom = (med[t]+a[t])*np.random.chisquare(N_tom-1)/(N_tom -1) - med[t]
            r.append(a_tom)
    else:
        for t in range(len(a)):
            a_tom = a[t]+(a[t]+med[t])*np.random.randn()*np.sqrt(2/(N_tom-1))
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

def get_moment_step_switching(wD,mcc,lam,LatLen,sigma,B,D,T,Tmin,Tmax,Ntimes,Ntom,smearing='gaussian',cutoff='none',eps=None, TOM=True):
    Ham, HE_thermal = ComputeHams(wD,mcc,lam,LatLen,sigma,B,D,smearing,cutoff,eps)
    ProjList = DefProjList(Ham, Ntimes, Tmin, Tmax)

    RS0 = InitializeProbeState('Ground')
    RE0 = ThermalState(HE_thermal, T)   # Compute the environment's thermal state
    RSE0 = directsum(RE0, RS0)             # Compute the initial probe-environment state
    aS = np.zeros(3*Ntimes)
    for k, P in enumerate(ProjList):
        for j, _P in enumerate(P):
            aS[3*k + j] = np.trace(_P @ RSE0)
    
    if TOM:
        aS_tom = Tomography(aS,Ntom,np.zeros(aS.shape))                            # Add tomographic noise
        return aS_tom
    else:
        return aS

def get_unitless_params(wD,sigma,mcc,lam,Tmin,Tmax,T,E0,T0,t0,a0,L):
    sigma *= 1/a0
    mcc   *= 1/E0
    wD    *= 1/E0
    lam   *= 1/E0
    T     *= 1/T0
    Tmin  *= 1/t0
    Tmax  *= 1/t0
    LatLen = int(L/a0)

    return wD, sigma, mcc, lam, LatLen, Tmin, Tmax, T