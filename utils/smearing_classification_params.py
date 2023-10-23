#! /usr/bin/python
import numpy as np
from scipy.constants import c, hbar, elementary_charge, Boltzmann

### Meta Parameters ###
experiment_name = "smearing_classification_run2_T=1mK_const_dt"
n_samples    = 10000              # Number of examples to produce for training/validating
Regression = False
overwrite = False

### Parameters of the quantum field ###
### Parameters of the quantum field ###
sigma = 53e-12                 # Detector Smearing [m]
Ksig = 16/sigma                    # Determines UV Cutoff [m^-1]
a = np.pi/Ksig      # Lattice spacing induced by UV Cutoff [m]
latlen = 457         # Determines IR Cutoff [a]
mcc = elementary_charge                     # Field mass [eV]
wD = 130*elementary_charge                    # Detector gap [eV]
lam = 130*elementary_charge                   # Coupling energy [eV]
Tmean = 1e-3


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
Tmean = Tmean/Temp0  # convert to hbar = c = a = k_b = 1 units
Tdev = 0*Tmean             # Size of Temperature range

### Measurement Options ###
dt= 1e-22             # Duration of each measurement window [s]
time_min = -21           # Start of first measurement window   [s]
time_max = -16      # End of last measurement window      [s]

plot_times_max = np.logspace(time_min,time_max,40,endpoint=True) # Linearly spaces measurement windows
plot_times_max = plot_times_max/T0
dt = dt/T0
plot_times_min = plot_times_max - dt # Linearly spaces measurement windows

print(Temp0)

measurements_per_window = 10 			          # Number of measurement times considered in each window
n_tom = 1e22                    # Number of times to repeat the whole experiment

### Defining Classes for Classification ###
TDev = 0             # Size of Temperature range

                                             
### Setting Active Cases (Label, Probability, Y-label, Details) ###
LPYD = np.array([
        ['Case Name',  'Abv'  , 'Prob', 'y', 'Boundary Type','Distance to Boundary','Temperature', 'Smearing', 'Dim'],
        ['Gaussian',   'Gau'  ,      1,   0,               1,         latlen,               Tmean, 'gaussian', 1],
        ['Lorentzian', 'Lor'  ,      1,   1,               1,         latlen,               Tmean, 'lorentzian', 1],
        ['Quartic'   , 'Qtc'  ,      1,   2,               1,         latlen,               Tmean, 'quartic', 1],
        ['Sharp'   ,   'Shp'  ,      1,   3,               1,         latlen,               Tmean, 'sharp', 1],
        ['Linear',     'Lin'  ,      0,   0,               1,         latlen,                   0, 'gaussian', 1],
        ['Circular',   'Cir'  ,      0,   1,               4,  int(latlen/2),                   0, 'gaussian', 1],
        ])

# Notes for smearing classification:
# take 0.95 variance, no middle later, 100 epochs, time from 1e-21 to 1e-17, 10000 samples
# Temperatures should be on the order of 1e-9, 457 things, 
# sigma = 53e-12                 # Detector Smearing [m]
# Ksig = 16/sigma                    # Determines UV Cutoff [m^-1]
# a = np.pi/Ksig      # Lattice spacing induced by UV Cutoff [m]
# latlen = 457         # Determines IR Cutoff [a]
# mcc = elementary_charge                     # Field mass [eV]
# wD = 130*elementary_charge                    # Detector gap [eV]
# lam = 130*elementary_charge                   # Coupling energy [eV]
# Tmean = 1e-7



### PCA Options ###
RunPCAonData = True               # Whether or not to do PCA
PCA_var_keep = 0.95                  # Fraction of variance to be kept after PCA (0 to 1 or 'All') 
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
L2reg = 0.01                    # L2 Regularizer
learning_rate = 0.001             # Learning Rate
n_epochs = 100                  # Number of epoch to train over
minibatch_size = 256             # Minibatch size
