#! /usr/bin/python
import numpy as np
from scipy.constants import c, hbar, elementary_charge, Boltzmann

### Meta Parameters ###
n_samples    = 10000              # Number of examples to produce for training/validating
Regression = False
overwrite = False
oned = True
gather_all_data_before_training = False
rerun_datapoints = True
reruns = [37, 38, 39]

### Parameters of the quantum field in 1D ###
exp1d = "circular_boundary_classification"
sigma = 18e-3                 # Detector Smearing [m]
Ksig = 16/sigma                    # Determines UV Cutoff [m^-1]
a = np.pi/Ksig      # Lattice spacing induced by UV Cutoff [m]
latlen = 100         # Determines IR Cutoff [a]
mcc = 0.1*1e9*hbar                     # Field mass [eV]
wD = 10*1e9*hbar                    # Detector gap [eV]
lam = 10*1e9*hbar                   # Coupling energy [eV]
Tmean = 0

### Parameters of the quantum field in 2D ###
exp2d = "2D_geometry_classification"
#Ksig2d = 5/sigma                    # Determines UV Cutoff [m^-1]
latlen2d = 25         # Determines IR Cutoff [a]


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

print(sigma, mcc, wD, lam, Tmean, a, a/c, a*latlen/c)

### Measurement Options ###
time_min = -12           # Start of first measurement window   [s]
time_max = -9      # End of last measurement window      [s]


plot_times_max = np.logspace(time_min, time_max, 41, endpoint=True) # Linearly spaces measurement windows
plot_times_max = plot_times_max/T0
plot_times_min = plot_times_max*plot_times_max[0]/plot_times_max[1]

print(plot_times_max)

measurements_per_window = 16    # Number of measurement times considered in each window
n_tom = 1e4                     # Number of times to repeat the whole experiment

### Defining Classes for Classification ###
TDev = 0             # Size of Temperature range

                                             
### Setting Active Cases (Label, Probability, Y-label, Details) ###
LPYD = np.array([
        ['Case Name',  'Abv'  , 'Prob', 'y', 'Boundary Type','Distance to Boundary','Temperature', 'Smearing', 'Dim', 'Lattice Length'],
        ['Linear end', 'LinEnd',     0,   0,               1,         latlen,               Tmean, 'gaussian',     1, latlen],
        ['Linear mid', 'LinMin',     1 if oned else 0,   0,               1,    latlen//2+1,               Tmean, 'gaussian',     1, latlen],
        ['Circular',   'Cir'  ,      1 if oned else 0,   1,               4,    latlen//2+1,               Tmean, 'gaussian',     1, latlen],
        ['2D_rect',    '2DR',        1 if not oned else 0,   0,               1,    latlen2d//2+1,             Tmean, 'gaussian',     2, latlen2d],
        ['2D_torus',   '2DT',        1 if not oned else 0,   1,               4,    latlen2d//2+1,             Tmean, 'gaussian',     2, latlen2d],
        ['Linear mid', 'LinMin',     0,   0,               1,    latlen//2+1,               Tmean, 'gaussian',     1, latlen],
        ['Linear mid', 'LinMin',     0,   1,               1,    (latlen+2)//2+1,           Tmean, 'gaussian',     1, latlen+2]

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

nH1 = 20                         # Number of neurons in the first hidden layer
L2reg = 1e-2                    # L2 Regularizer
learning_rate = 1e-2             # Learning Rate
n_epochs = 10                  # Number of epoch to train over
minibatch_size = 256             # Minibatch size

experiment_name = (exp1d if oned else exp2d) + \
                  "_ntom=1e" + str(int(np.log10(n_tom))).replace('.', 'p') + \
                  "_T=" + str(Tmean*Temp0).replace('.', 'p') + \
                  "K_nH1=" + str(nH1) + \
                  "_n_epochs=" + str(n_epochs) + \
                  "_l2reg=1e" + str(np.log10(L2reg)).replace('.', 'p') + \
                  "_lr=1e" +str(np.log10(learning_rate)).replace('.', 'p')