#! /usr/bin/python
import numpy as np
from scipy.constants import c, hbar, elementary_charge, Boltzmann

### Meta Parameters ###
N = 4
CaseString = 'Fock_vs_PAC_'+str(N)         # String Associated with this run of the code
GenerateData = True             # If False, code will read in old data
N_Samples    = 4000              # Number of examples to produce for training/validating
Regression = False

### Units used are hbar = c = a = 1 ###
q = elementary_charge

### Parameters of the quantum field ###
L = 0.01 # [m]
sigma = 1e-4 # [m]
m = 0
Ksig = 7/sigma                    # Determines UV Cutoff [m^-1]
a = np.pi/Ksig
N_cutoff = int(Ksig/(np.pi/L))
E0 = hbar*c/(q*a)   # Energy unit

### Measurement Options ###
Tmin = 0           # Start of first measurement window   [s]
Tmax = 66.7*1e-12      # End of last measurement window      [s]

### Unitless Parameters ###
sigma = sigma/a            # Normalized Smearing
L = L/a            # Normalized Smearing

# convert to units k_b = hbar = c = a = 1
Tmin *= c/a
Tmax *= c/a

wDmin = 10*1e9
wDmax = 220*1e9
dw = 10*1e9

wD_list = np.linspace(wDmin, wDmax, int((wDmax-wDmin)/dw)+1) # Linearly spaces measurement windows
wD_list *= a/c
wD_list = list(set(wD_list))                               # Removes duplicates
wD_list.sort()                                               # Sorts list

N_times = 2 			          # Number of measurement times considered in each window
N_tom = 1e15                    # Number of times to repeat the whole experiment

### Defining Classes for Classification ###

### Setting Active Cases (Label, Probability, Y-label, Details) ###
LPYD = np.array([
        ['Case Name',  'Abv'  , 'Prob', 'y'],
        ['FOCK',       'fock',      1,    1],
        ['PAC'  ,      'pac'  ,     1,    2],
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

### PCA Options ###
RunPCAonData = True               # Whether or not to do PCA
PCA_Var_Keep = 1                  # Fraction of variance to be kept after PCA (0 to 1 or 'All') 
N_PCA_plot   = 1000               # Number of data points to plotted in PCA 

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
N_epochs = 21                  # Number of epoch to train over
minibatch_size = 100             # Minibatch size

