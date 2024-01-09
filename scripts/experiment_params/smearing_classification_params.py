#! /usr/bin/python
import numpy as np
from scipy.constants import c, hbar, elementary_charge, Boltzmann

### Meta Parameters ###
n_samples    = 10000              # Number of examples to produce for training/validating
Regression = False
overwrite = False
gather_all_data_before_training = False
rerun_datapoints = False
reruns = list(range(35, 52))

### Parameters of the quantum field ###
sigma = 18e-3                 # Detector Smearing [m]
Ksig = 16/sigma                    # Determines UV Cutoff [m^-1]
a = np.pi/Ksig      # Lattice spacing induced by UV Cutoff [m]
latlen = 100         # Determines IR Cutoff [a]
mcc = 0.1*1e9*hbar                     # Field mass [eV]
wD = 10*1e9*hbar                    # Detector gap [eV]
lam = 10*1e9*hbar                   # Coupling energy [eV]
Tmean = 10


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
time_min = -12           # Start of first measurement window   [s]
time_max = -6      # End of last measurement window      [s]
n_windows = 61

plot_times_max = np.logspace(time_min,time_max,n_windows,endpoint=True) # Linearly spaces measurement windows
plot_times_max = plot_times_max/T0
plot_times_min = plot_times_max*plot_times_max[0]/plot_times_max[1]


measurements_per_window = 32 			          # Number of measurement times considered in each window
n_tom = 1e7                    # Number of times to repeat the whole experiment

### Defining Classes for Classification ###
TDev = 0             # Size of Temperature range

                                             
### Setting Active Cases (Label, Probability, Y-label, Details) ###
LPYD = np.array([
        ['Case Name',  'Abv'  , 'Prob', 'y', 'Boundary Type','Distance to Boundary','Temperature', 'Smearing', 'Dim'],
        ['Gaussian',   'Gau'  ,      1,   0,               1,  int(latlen/2),               Tmean, 'gaussian', 1],
        ['Lorentzian', 'Lor'  ,      1,   1,               1,  int(latlen/2),               Tmean, 'lorentzian', 1],
        ['Quartic'   , 'Qtc'  ,      1,   2,               1,  int(latlen/2),               Tmean, 'quartic', 1],
        ['Sharp'   ,   'Shp'  ,      1,   3,               1,  int(latlen/2),               Tmean, 'sharp', 1],
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
L2reg = 1e-2                    # L2 Regularizer
learning_rate = 1e-3             # Learning Rate
n_epochs = 10                 # Number of epoch to train over
minibatch_size = 256             # Minibatch size

experiment_name = "smearing_classification" + \
                  "_ntom=1e" + str(int(np.log10(n_tom))).replace('.', 'p') + \
                  "_T=" + str(Tmean*Temp0).replace('.', 'p') + \
                  "K_nH1=" + str(nH1) + \
                  "_n_epochs=" + str(n_epochs) + \
                  "_l2reg=1e" + str(np.log10(L2reg)).replace('.', 'p') + \
                  "_lr=1e" +str(np.log10(learning_rate)).replace('.', 'p')