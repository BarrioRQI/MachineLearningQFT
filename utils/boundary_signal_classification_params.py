#! /usr/bin/python
import numpy as np
from scipy.constants import c, hbar, elementary_charge, Boltzmann

### Meta Parameters ###
TMean = '350_uK_FINAL'
CaseString = 'TEST'+str(TMean)         # String Associated with this run of the code
GenerateData = True             # If False, code will read in old data
n_samples    = 11250              # Number of examples to produce for training/validating
Regression = False

### Parameters of the quantum field ###
### Parameters of the quantum field ###
sigma = 53e-12                 # Detector Smearing [m]
Ksig = 16/sigma                    # Determines UV Cutoff [m^-1]
a = np.pi/Ksig      # Lattice spacing induced by UV Cutoff [m]
latlen = 457         # Determines IR Cutoff [a]
mcc = elementary_charge                     # Field mass [eV]
wD = 130*elementary_charge                    # Detector gap [eV]
lam = 130*elementary_charge                   # Coupling energy [eV]
Tmean = 0

### Measurement Options ###
dt= 1*1e-18             # Duration of each measurement window [s]
time_min = 13*1e-18           # Start of first measurement window   [s]
time_max = 17*1e-18      # End of last measurement window      [s]

### Units used are hbar = c = a = 1 ###
E0 = hbar*c/a   # Energy unit
Temp0 = a*Boltzmann/(hbar*c)   # Energy unit
L0 = a   # Energy unit
T0 = a/c   # Energy unit

### Unitless Parameters ###
sigma = sigma/L0            # Normalized Smearing
mcc = mcc/E0               # Normalized Field Mass
wD = wD/E0                 # Normalized Detector Gap   
lam = lam/E0               # Normalized Coupling Energy
tmin = time_min/T0
tmax = time_max/T0
dt = dt/T0
Tmean = Tmean/Temp0  # convert to hbar = c = a = k_b = 1 units
Tdev = 0*Tmean             # Size of Temperature range


plot_times = list(np.linspace(tmin,tmax,int((tmax-tmin)/dt)+1)) # Linearly spaces measurement windows
#PlotTimes += list(np.linspace(0.8,3.2,13)) # Linearly spaces measurement windows
plot_times = list(set(plot_times))                               # Removes duplicates
plot_times.sort()                                               # Sorts list
measurements_per_window = 10 			          # Number of measurement times considered in each window
n_tom = 1e22                    # Number of times to repeat the whole experiment

### Defining Classes for Classification ###
# Thermal Parameters
TMean = (350*1e-6)*(Boltzmann*a)/(hbar*c)  # convert to hbar = c = a = k_b = 1 units
TDev = 0.0*TMean             # Size of Temperature range

# Bounsary Sensing Parameters
Gsignal = 3.5143             # Squeezing of the "signal oscillator" 8db -> r = 1/2e^(0.8)
                                             
### Setting Active Cases (Label, Probability, Y-label, Details) ###
LPYD = np.array([
        ['Case Name',  'Abv'  , 'Prob', 'y', 'Boundary Type','Distance to Boundary','Temperature', 'Smearing', 'Dim'],
        ['Full Bond',  'Uncut',      0,   1,               1,         latlen,            0, 'gaussian', 1],
        ['No Bond'  ,  'Cut'  ,      1,   2,               2,         latlen,            0, 'gaussian', 1],
        ['Signal'   ,  'Sign' ,      1,   3,               3,         latlen,            0, 'gaussian', 1],
        ['89-91%'   ,  '0.9'  ,      0,   0,               1,  int(latlen/2),   0.90*TMean, 'gaussian', 1],
        ['91-93%'   ,  '0.92' ,      0,   1,               1,  int(latlen/2),   0.92*TMean, 'gaussian', 1],
        ['93-95%'   ,  '0.94' ,      0,   2,               1,  int(latlen/2),   0.94*TMean, 'gaussian', 1],
        ['95-97%'   ,  '0.96' ,      0,   3,               1,  int(latlen/2),   0.96*TMean, 'gaussian', 1],
        ['97-99%'   ,  '0.98' ,      0,   4,               1,  int(latlen/2),   0.98*TMean, 'gaussian', 1],
        ['99-101%'  ,  '1.00' ,      0,   5,               1,  int(latlen/2),   1.00*TMean, 'gaussian', 1],
        ['101-103%' ,  '1.02' ,      0,   6,               1,  int(latlen/2),   1.02*TMean, 'gaussian', 1],
        ['103-105%' ,  '1.04' ,      0,   7,               1,  int(latlen/2),   1.04*TMean, 'gaussian', 1],
        ['105-107%' ,  '1.06' ,      0,   8,               1,  int(latlen/2),   1.06*TMean, 'gaussian', 1],
        ['107-109%' ,  '1.08' ,      0,   9,               1,  int(latlen/2),   1.08*TMean, 'gaussian', 1],
        ['109-111%' ,  '1.10' ,      0,  10,               1,  int(latlen/2),   1.10*TMean, 'gaussian', 1]
        ])

### PCA Options ###
RunPCAonData = True               # Whether or not to do PCA
PCA_var_keep = 1                  # Fraction of variance to be kept after PCA (0 to 1 or 'All') 
N_PCA_plot   = 1000               # Number of data points to ploted in PCA 

### Neural Network Options ###
RunNNonData = True                # Whether or not to train the Neural Network
f_train = 75                      # Fraction of data used for training 
f_valid = 20                      # Fraction of data used for validation
f_test = 5                        # Fraction of data reserved for testing
fsum = f_train + f_valid + f_test # Normalize
f_train = f_train/fsum
f_valid = f_valid/fsum
f_test  = f_test/fsum

nH1 = 30                         # Number of neurons in the first hidden layer
L2reg = 0.001                    # L2 Regularizer
learning_rate = 0.01             # Learning Rate
n_epochs = 50                  # Number of epoch to train over
minibatch_size = 100             # Minibatch size

