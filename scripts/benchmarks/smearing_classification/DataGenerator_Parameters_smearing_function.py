#! /usr/bin/python
import numpy as np
from scipy.constants import c, hbar, elementary_charge, Boltzmann

k = 16*4

### Meta Parameters ###
CaseString = 'all_classification_K_'+str(k)         # String Associated with this run of the code
GenerateData = True             # If False, code will read in old data
N_Samples    = 5000              # Number of examples to produce for training/validating
Regression = False
save_data = True
save_intermediary_data = False

### Units used are hbar = c = a = 1 ###
q = elementary_charge

### Parameters of the quantum field ###
sigma = 10*1e-6 # [m]
wD = 10*1e9
lam = 10*1e9
m = 0.1*1e9
T = 0
L = 0.5e-3
eps = 50*1e9

# unitless params

K = k*c/sigma            # Determines UV Cutoff [m^-1]


if c/sigma < wD + eps:
    K = k*(wD + eps)
else:
    K = k*c/sigma

a = np.pi/K * c
LatticeLen = int(np.around(L/a))
print(LatticeLen)
E0 = hbar/q                 # Energy unit
t0 = a/c                  # Time unit
a0 = a                    # distance unit


wD *= t0
eps *= t0
lam *= t0
m *= t0
sigma /= a0

# N times doesn't matter, N_tom matters until a threshold
# Tmin and Tmax matters a LOT


### measurement options ###
N_times = 20
N_tom = 1e30

### Measurement Options ###
Tmin = -20                  # Start of first measurement window   [s]
Tmax = 0         # End of last measurement window      [s]
dt   = 0.5         # End of last measurement window      [s]

MeasurementTimes = 10**(np.linspace(Tmin, Tmax, int((Tmax-Tmin)/dt)+1)) # Linearly spaces measurement windows
MeasurementTimes = list(set(MeasurementTimes))                                                     # Removes duplicates
MeasurementTimes.sort()                                                                       # Sorts list
MeasurementTimes = np.asarray(MeasurementTimes)/t0

### Setting Active Cases (Label, Probability, Y-label, Details) ###

LPYD = np.array([
        ['Case',  'Abv'  ,  'Prob', 'y', 'Mass', 'smearing', 'cutoff', 'BT', 'D2B',  'smearing type',  'cutoff type'],
        ['GG'  ,   'GG'  ,       1,   0,  m,      sigma,          eps,    1,     0,     'gaussian',       'gaussian'],
        ['GL'  ,   'GL'  ,       1,   1,  m,      sigma,          eps,    1,     0,     'gaussian',     'lorentzian'],
        ['GQ'  ,   'GQ'  ,       1,   2,  m,      sigma,          eps,    1,     0,     'gaussian',    'exponential'],
        ['GS'  ,   'GS'  ,       1,   3,  m,      sigma,          eps,    1,     0,     'gaussian',          'sharp'],
        ['LG'  ,   'LG'  ,       1,   4,  m,      sigma,          eps,    1,     0,   'lorentzian',       'gaussian'],
        ['LL'  ,   'LL'  ,       1,   5,  m,      sigma,          eps,    1,     0,   'lorentzian',     'lorentzian'],
        ['LE'  ,   'LE'  ,       1,   6,  m,      sigma,          eps,    1,     0,   'lorentzian',    'exponential'],
        ['LS'  ,   'LS'  ,       1,   7,  m,      sigma,          eps,    1,     0,   'lorentzian',          'sharp'],
        ['QG'  ,   'QG'  ,       1,   8,  m,      sigma,          eps,    1,     0,      'quartic',       'gaussian'],
        ['QL'  ,   'QL'  ,       1,   9,  m,      sigma,          eps,    1,     0,      'quartic',     'lorentzian'],
        ['QE'  ,   'QE'  ,       1,  10,  m,      sigma,          eps,    1,     0,      'quartic',    'exponential'],
        ['QS'  ,   'QS'  ,       1,  11,  m,      sigma,          eps,    1,     0,      'quartic',          'sharp'],
        ['SG'  ,   'SG'  ,       1,  12,  m,      sigma,          eps,    1,     0,        'sharp',       'gaussian'],
        ['SL'  ,   'SL'  ,       1,  13,  m,      sigma,          eps,    1,     0,        'sharp',     'lorentzian'],
        ['SE'  ,   'SE'  ,       1,  14,  m,      sigma,          eps,    1,     0,        'sharp',    'exponential'],
        ['SS'  ,   'SS'  ,       1,  15,  m,      sigma,          eps,    1,     0,        'sharp',          'sharp'],
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
Clist = LPYD[:,6].astype(float)    # Extract boundary conditions from LPYD
Blist = LPYD[:,7].astype(int)    # Extract boundary conditions from LPYD
Dlist = LPYD[:,8].astype(int)    # Extract boundary conditions from LPYD
Smearinglist = LPYD[:,9]    # Extract boundary conditions from LPYD
CutoffList = LPYD[:,10]    # Extract boundary conditions from LPYD

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
N_epochs = 200                    # Number of epoch to train over
minibatch_size = 100             # Minibatch size
verbose = 20