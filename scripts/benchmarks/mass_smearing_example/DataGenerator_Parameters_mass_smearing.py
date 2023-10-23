#! /usr/bin/python
import numpy as np
from scipy.constants import c, hbar, elementary_charge, Boltzmann

### Meta Parameters ###
CaseString = 'mass_regression_FINALFINAL'         # String Associated with this run of the code
GenerateData = True             # If False, code will read in old data
N_Samples    = 300              # Number of examples to produce for training/validating
Regression = True
save_data = True
save_intermediary_data = False

### Units used are hbar = c = a = 1 ###
q = elementary_charge

### Parameters of the quantum field ###
sigma = 18e-3 # [m]
wD = 10*1e9*hbar/q
lam = 10*1e9*hbar/q
m = 0.1*1e9*hbar/q
T = 200e-6
L = 0.353

def get_natural_units(sigma):
    ### unit conversion ###
    Ksig = 16/sigma                    # Determines UV Cutoff [m^-1]
    a = np.pi/Ksig

    E0 = hbar*c/(q*a)         # Energy unit
    T0 = hbar*c/(Boltzmann*a) # Temperature unit
    t0 = a/c                  # Time unit
    a0 = a                    # distance unit

    return E0, T0, t0, a0

### measurement options ###
N_times_min = 2
N_times_max = 16
dN = 1
N_tom = 1e22

Ntimes_list = np.linspace(N_times_min, N_times_max, int((N_times_max-N_times_min)/dN)+1) # Linearly spaces measurement windows
Ntimes_list = np.array(Ntimes_list, dtype=int)
Ntimes_list = list(set(Ntimes_list))                                                     # Removes duplicates
Ntimes_list.sort()                                                                       # Sorts list


### Measurement Options ###
Tmin = -17                  # Start of first measurement window   [s]
Tmax = -8         # End of last measurement window      [s]
dt   = 1         # End of last measurement window      [s]

MeasurementTimes = 10**(np.linspace(Tmin, Tmax, int((Tmax-Tmin)/dt)+1)) # Linearly spaces measurement windows
MeasurementTimes = list(set(MeasurementTimes))                                                     # Removes duplicates
MeasurementTimes.sort()                                                                       # Sorts list

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

E0, T0, t0, a0 = get_natural_units(sigma)
wD_u, sigma_u, mcc_u, lam_u, LatLen, Tmin_u, Tmax_u, T_u = \
    get_unitless_params(wD,sigma,m,lam,Tmin,Tmax,T,E0,T0,t0,a0,L)
print(wD_u, sigma_u, mcc_u, lam_u, LatLen, Tmin_u, Tmax_u, T_u)

### Setting Active Cases (Label, Probability, Y-label, Details) ###
MDev = 0.01*m
LPYD = np.array([
        ['Case'     ,  'Abv'  ,  'Prob', 'y', 'Mass', 'Smearing',  'Boundary Type', 'Dist to Boundary'],
        ['mass-5'   ,   '-5'  ,       1,   0,  0.95*m,      sigma,                1,     0],
        ['mass-4'   ,   '-4'  ,       1,   1,  0.96*m,      sigma,                1,     0],
        ['mass-3'   ,   '-3'  ,       1,   2,  0.97*m,      sigma,                1,     0],
        ['mass-2'   ,   '-2'  ,       1,   3,  0.98*m,      sigma,                1,     0],
        ['mass-1'   ,   '-1'  ,       1,   4,  0.99*m,      sigma,                1,     0],
        [ 'mass0'   ,    '0'  ,       1,   5,  1.00*m,      sigma,                1,     0],
        [ 'mass1'   ,    '1'  ,       1,   6,  1.01*m,      sigma,                1,     0],
        [ 'mass2'   ,    '2'  ,       1,   7,  1.02*m,      sigma,                1,     0],
        [ 'mass3'   ,    '3'  ,       1,   8,  1.03*m,      sigma,                1,     0],
        [ 'mass4'   ,    '4'  ,       1,   9,  1.04*m,      sigma,                1,     0],
        [ 'mass5'   ,    '5'  ,       1,  10,  1.05*m,      sigma,                1,     0],
        ])

LPYD = LPYD[1:,:]                            # Remove first row 
nz = np.nonzero(LPYD[:,2].astype(float))[0]  # Find rows with non-zero probability
LPYD = LPYD[nz,:]                            # Isolate Rows with non-zero probability

DupYLabels = LPYD[:,1]          # Extract the y labels from LPYD (with duplicates)
YLabels = []                    # Remove the duplicates
for i in DupYLabels: 
    if i not in YLabels: 
        YLabels.append(i)

plist = LPYD[:,2].astype(float) # Extract probabilities from LYPD
ptot  = sum(plist)              # Compute their sum
plist = plist/ptot              # Normalize them
Cases = len(plist)              # Number of cases being considered

ylist  = LPYD[:,3].astype(int)  # Extract y-values from LPYD (with duplicates)
ylist2 = []                     # Remove duplicates from ylist
for i in ylist: 
    if i not in ylist2: 
        ylist2.append(i)    
for k in range(len(ylist)):     # Reduce values in ylist to 0 through Cases
    ylist[k] = ylist2.index(ylist[k])
ylist2 = str(ylist2)

Mlist = LPYD[:,4].astype(float)      # Extract mass from LPYD
Slist = LPYD[:,5].astype(float)    # Extract smearing from LPYD
Blist = LPYD[:,6].astype(int)    # Extract boundary conditions from LPYD
Dlist = LPYD[:,7].astype(int)    # Extract boundary conditions from LPYD

### PCA Options ###
RunPCAonData = True               # Whether or not to do PCA
PCA_Var_Keep = 1                  # Fraction of variance to be kept after PCA (0 to 1 or 'All') 
N_PCA_plot   = 800               # Number of data points to plotted in PCA 

### Neural Network Options ###
RunNNonData = True                # Whether or not to train the Neural Network
f_train = 75                      # Fraction of data used for training 
f_valid = 25                      # Fraction of data used for validation
f_test = 0                        # Fraction of data reserved for testing
fsum = f_train + f_valid + f_test # Normalize
f_train = f_train/fsum
f_valid = f_valid/fsum
f_test  = f_test/fsum

nH1 = 30                         # Number of neurons in the first hidden layer
L2reg = 0.001                    # L2 Regularizer
learning_rate = 0.01             # Learning Rate
N_epochs = 1000                    # Number of epoch to train over
minibatch_size = 100             # Minibatch size
verbose = 100