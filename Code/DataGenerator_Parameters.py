import numpy as np

### Meta Parameters ###
CaseString = 'RegStuff'     # String Associated with this run of the code
GenerateData = True             # If False, code will read in old data
N_Samples    = 200              # Number of examples to produce for training/validating
Regression = False

### Parameters of the quantum field ###
sigma = 4.2                 # Detector Smearing
HbarCbySig = 7.14           # Energy of Smearing = Hbar * c / sigma 
Ksig = 7                    # Determines UV Cutoff
a = np.pi/Ksig * sigma      # Lattice spacing induced by UV Cutoff
LatticeLength = 10         # Determines IR Cutoff
mcc = 0.1                   # Field mass
wD = 10                     # Detector gap
lam = 10                    # Coupling energy     
    
### Units used are hbar = c = a = 1 ###
E0 = HbarCbySig * Ksig/np.pi   # Energy unit
a0 = a                         # Distance unit

### Unitless Parameters ###
sigma = sigma/a0           # Normalized Smearing
mcc = mcc/E0               # Normalized Field Mass
wD = wD/E0                 # Normalized Detector Gap   
lam = lam/E0               # Normaalized Coupling Energy

### Measurement Options ###
dt= 0.4             # Duration of each measurement window
Tmin = 2.8            # Start of first measurement window
Tmax = 3.2          # End of last measurement window
PlotTimes = list(np.linspace(Tmin,Tmax,int((Tmax-Tmin)/dt)+1)) # Linearly spaces measurement windows
#PlotTimes += list(np.linspace(0.8,3.2,13)) # Linearly spaces measurement windows
PlotTimes = list(set(PlotTimes))                               # Removes duplicates
PlotTimes.sort()                                               # Sorts list
N_times = 10 			          # Number of measurement times considered in each window
N_tom = 10**20                    # Number of times to repeat the whole experiment

### Defining Classes for Classification ###
# Thermal Parameters
TMean = 1/127             # Mean Temperature for Thermal Cases
TDev = 0.01*TMean             # Size of Temperature Bins

# Bounsary Sensing Parameters
Gsignal = 3.1548              # Squeezing of the "signal oscillator" 8db -> r = 1/2e^(0.8)
                                             
### Setting Active Cases (Label, Probability, Y-label, Details) ###
LPYD = np.array([
        ['Case Name',  'Abv'  , 'Prob', 'y', 'Boundary Type','Distance to Boundary','Temperature'],
        ['Full Bond',  'Uncut',      0,   1,               1,         LatticeLength,            0],
        ['No Bond'  ,  'Cut'  ,      0,   2,               2,         LatticeLength,            0],
        ['Signal'   ,  'Sign' ,      0,   3,               3,         LatticeLength,            0],
        ['89-91%'   ,  '0.9'  ,      1,   0,               1,  int(LatticeLength/2),   0.90*TMean],
        ['91-93%'   ,  '0.92' ,      1,   1,               1,  int(LatticeLength/2),   0.92*TMean],
        ['93-95%'   ,  '0.94' ,      1,   2,               1,  int(LatticeLength/2),   0.94*TMean],
        ['95-97%'   ,  '0.96' ,      1,   3,               1,  int(LatticeLength/2),   0.96*TMean],
        ['97-99%'   ,  '0.98' ,      1,   4,               1,  int(LatticeLength/2),   0.98*TMean],
        ['99-101%'  ,  '1.00' ,      1,   5,               1,  int(LatticeLength/2),   1.00*TMean],
        ['101-103%' ,  '1.02' ,      1,   6,               1,  int(LatticeLength/2),   1.02*TMean],
        ['103-105%' ,  '1.04' ,      1,   7,               1,  int(LatticeLength/2),   1.04*TMean],
        ['105-107%' ,  '1.06' ,      1,   8,               1,  int(LatticeLength/2),   1.06*TMean],
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
L2reg = 0.0001                        # L2 Regularizer
learning_rate = 0.01           # Learning Rate
N_epochs = 1000                   # Number of epoch to train over
minibatch_size = 100             # Minibatch size

