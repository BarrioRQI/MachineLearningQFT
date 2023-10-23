#! /usr/bin/python
import numpy as np
from scipy.constants import c, hbar, elementary_charge, Boltzmann

### Meta Parameters ###
#experiment_name = "test19"
experiment_name = "position_regression_T=0_L=100_sig=18e_3_ALL_PCA"
n_samples    = 10000              # Number of examples to produce for training/validating
Regression = True
overwrite = False

### Parameters of the quantum field ###
### Parameters of the quantum field ###
sigma = 50e-3                 # Detector Smearing [m]
Ksig = 16/sigma                    # Determines UV Cutoff [m^-1]
a = np.pi/Ksig      # Lattice spacing induced by UV Cutoff [m]
latlen = 100         # Determines IR Cutoff [a]
mcc = 0.1*1e9*hbar                     # Field mass [eV]
wD = 10*1e9*hbar                    # Detector gap [eV]
lam = 10*1e9*hbar                   # Coupling energy [eV]
Tmean = 100e-6


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
dt = 1e-9             # Duration of each measurement window [s]
time_min = -12           # Start of first measurement window   [s]
time_max = -5      # End of last measurement window      [s]


plot_times_max = np.logspace(-13,-8,50,endpoint=True) # Linearly spaces measurement windows
plot_times_max = plot_times_max/T0
plot_times_min = np.zeros(plot_times_max.shape) # Linearly spaces measurement windows

#plot_times_max = np.linspace(5e-16,1e-14,100,endpoint=True) # Linearly spaces measurement windows
#plot_times_max = plot_times_max/T0
#dt = plot_times_max[0]
#plot_times_min = plot_times_max - dt # Linearly spaces measurement windows

measurements_per_window = 10 			          # Number of measurement times considered in each window
n_tom = 1e22                    # Number of times to repeat the whole experiment

### Defining Classes for Classification ###
TDev = 0             # Size of Temperature range

                                             
### Setting Active Cases (Label, Probability, Y-label, Details) ###
LPYD = [['Case Name',  'Abv'  , 'Prob', 'y', 'Boundary Type','Distance to Boundary','Temperature', 'Smearing', 'Dim'],]

latmid = latlen//2

for i in range(latlen//2):
    LPYD.append([str(i),str(i), 1, i/latmid + 1/(2*latmid), 1, i, Tmean, 'gaussian', 1])


LPYD = np.asarray(LPYD, dtype=object)
LPYD[1:, 3] = np.around(np.asarray(LPYD[1:, 3], dtype=float), 3)
#print(LPYD)

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
n_epochs = 10                  # Number of epoch to train over
minibatch_size = 100             # Minibatch size
