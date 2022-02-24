#! /usr/bin/python
import numpy as np
from scipy.constants import c, hbar, elementary_charge, Boltzmann

### Meta Parameters ###
TMean = '350_uK_FINAL'
CaseString = 'TEST'+str(TMean)         # String Associated with this run of the code
GenerateData = True             # If False, code will read in old data
N_Samples    = 30              # Number of examples to produce for training/validating
Regression = False

### Parameters of the quantum field ###
sigma = 42e-3                 # Detector Smearing [m]

Ksig = 7/sigma                    # Determines UV Cutoff [m^-1]
a = np.pi/Ksig      # Lattice spacing induced by UV Cutoff [m]
LatticeLength = 100         # Determines IR Cutoff [a]
mcc = (1e8)*(1.054*1e-34)/(1.6*1e-19)                     # Field mass [eV]
wD = (1e10)*(1.054*1e-34)/(1.6*1e-19)                    # Detector gap [eV]
lam = (1e10)*(1.054*1e-34)/(1.6*1e-19)                   # Coupling energy [eV]

### Units used are hbar = c = a = 1 ###
q = elementary_charge
E0 = hbar*c/(q*a)   # Energy unit

### Unitless Parameters ###
sigma = sigma/a            # Normalized Smearing
mcc = mcc/E0               # Normalized Field Mass
wD = wD/E0                 # Normalized Detector Gap   
lam = lam/E0               # Normalized Coupling Energy


### Measurement Options ###
dt= 10*1e-12             # Duration of each measurement window [s]
Tmin = 0*150*1e-12           # Start of first measurement window   [s]
Tmax = 200*1e-12      # End of last measurement window      [s]

# convert to units k_b = hbar = c = a = 1
Tmin *= c/a
Tmax *= c/a
dt *= c/a
print(sigma, mcc, wD, lam, a)

print(Tmin, Tmax, dt)

PlotTimes = list(np.linspace(Tmin,Tmax,int((Tmax-Tmin)/dt)+1)) # Linearly spaces measurement windows
#PlotTimes += list(np.linspace(0.8,3.2,13)) # Linearly spaces measurement windows
PlotTimes = list(set(PlotTimes))                               # Removes duplicates
PlotTimes.sort()                                               # Sorts list
N_times = 3 			          # Number of measurement times considered in each window
N_tom = 1e20                    # Number of times to repeat the whole experiment

### Defining Classes for Classification ###
# Thermal Parameters
TMean = (350*1e-6)*(Boltzmann*a)/(hbar*c)  # convert to hbar = c = a = k_b = 1 units
TDev = 0.01*TMean             # Size of Temperature range

# Bounsary Sensing Parameters
Gsignal = 3.5143             # Squeezing of the "signal oscillator" 8db -> r = 1/2e^(0.8)
                                             
### Setting Active Cases (Label, Probability, Y-label, Details) ###
LPYD = np.array([
        ['Case Name',  'Abv'  , 'Prob', 'y', 'Boundary Type','Distance to Boundary','Temperature'],
        ['Full Bond',  'Uncut',      0,   1,               1,         LatticeLength,            0],
        ['No Bond'  ,  'Cut'  ,      0,   2,               2,         LatticeLength,            0],
        ['Signal'   ,  'Sign' ,      0,   3,               3,         LatticeLength,            0],
        ['89-91%'   ,  '0.9'  ,      0,   0,               1,  int(LatticeLength/2),   0.90*TMean],
        ['91-93%'   ,  '0.92' ,      0,   1,               1,  int(LatticeLength/2),   0.92*TMean],
        ['93-95%'   ,  '0.94' ,      0,   2,               1,  int(LatticeLength/2),   0.94*TMean],
        ['95-97%'   ,  '0.96' ,      0,   3,               1,  int(LatticeLength/2),   0.96*TMean],
        ['97-99%'   ,  '0.98' ,      0,   4,               1,  int(LatticeLength/2),   0.98*TMean],
        ['99-101%'  ,  '1.00' ,      0,   5,               1,  int(LatticeLength/2),   1.00*TMean],
        ['101-103%' ,  '1.02' ,      0,   6,               1,  int(LatticeLength/2),   1.02*TMean],
        ['103-105%' ,  '1.04' ,      0,   7,               1,  int(LatticeLength/2),   1.04*TMean],
        ['105-107%' ,  '1.06' ,      0,   8,               1,  int(LatticeLength/2),   1.06*TMean],
        ['107-109%' ,  '1.08' ,      1,   9,               1,  int(LatticeLength/2),   1.08*TMean],
        ['109-111%' ,  '1.10' ,      1,  10,               1,  int(LatticeLength/2),   1.10*TMean]
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
    
Blist = LPYD[:,4].astype(int)      # Extract Boudary conditions from LPYD
Dlist = LPYD[:,5].astype(int)      # Extract Distances from LPYD
TempList = LPYD[:,6].astype(float) # Extract Distances from LPYD

### PCA Options ###
RunPCAonData = True               # Whether or not to do PCA
PCA_Var_Keep = 1                  # Fraction of variance to be kept after PCA (0 to 1 or 'All') 
N_PCA_plot   = 1000               # Number of data points to ploted in PCA 

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
N_epochs = 1000                  # Number of epoch to train over
minibatch_size = 100             # Minibatch size

