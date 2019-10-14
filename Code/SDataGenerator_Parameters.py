import numpy as np

### Meta Parameters ###
CaseString = 'ThermalStuff'
GenerateData = False
N_Samples    = 100                       # Number of scenarios to compute

### Parameters from quantum field
sigma = 4.2                 # Detector Smearing
HbarCbySig = 7.14           # Hbar * c / sigma 
Ksig = 7                    # UV Cutoff
a = np.pi/Ksig * sigma      # Lattice spacing
LatticeLength = 300          # IR Cutoff
mcc = 0.1                   # Field mass
wD = 10                     # Detector gap
lam =10
    
E0 = HbarCbySig * Ksig/np.pi    # Energy unit
a0 = a                          # Distance unit

sigma = sigma/a0           # Normalized smearing
wD = wD/E0                    # Normalized Detector Gap   
mcc = mcc/E0                 # Normalized Field Mass
lam = lam/E0

### Defining the Dynamics ###
dt= 0.4
Tmi = 0
Tma = 0.4
PlotTimes = list(np.linspace(Tmi,Tma,int(1+(Tma-Tmi)/dt)))
#+list(np.linspace(1.0,3.2,12))
#list([180,183.544,184.18,184.81,185.44,186.08,186.71,187.34,187.97,188.61,189.24,189.87])
PlotTimes = list(set(PlotTimes))
PlotTimes.sort()
N_times        = 10 			                # Number of time steps in each scenario

### Picking Environment States ###
ThermalStateDist = 'Uniform'                 # If 'Gaussian', T = Gaussian(mu,stdev)
                                             #    'Uniform' , T = Uniform(mu.stdev)
TMean = 0.125/127
TDev = 0*TMean                        # Std. Dev for above distributions
Gsignal = 3.1548                            # Squeezing of the "signal oscillator" 8db -> r = 1/2e^(0.8)

                                             
LPYD = np.array([
        ['Case Name','Y Name'   ,'p_c','y','B.Case',     'Dist'         , 'Temp'],
        ['Uncut'    ,'Full Bond',    0,  1,      1,     LatticeLength,      0],
        ['Cut'      ,'No Bond'  ,    0,  2,      2,     LatticeLength,      0],
        ['Signal'   ,'Signal'   ,    0,  3,      3,     LatticeLength,      0],
        ['Too cold' ,'0.9'      ,    1,  0,      1 ,    int(LatticeLength/2), 0.90*TMean],
        ['Cold'     ,'0.92'     ,    1,  1,      1 ,    int(LatticeLength/2), 0.92*TMean],
        ['Thermal'  ,'0.94'     ,    0,  2,      1 ,    int(LatticeLength/2), 0.94*TMean],
        ['Hot'      ,'0.96'     ,    0,  3,      1 ,    int(LatticeLength/2), 0.96*TMean],
        ['Too hot'  ,'0.98'     ,    0,  4,      1 ,    int(LatticeLength/2), 0.98*TMean],
        ['Too cold' ,'1.00'     ,    0,  5,      1 ,    int(LatticeLength/2), 1.00*TMean],
        ['Cold'     ,'1.02'     ,    0,  6,      1 ,    int(LatticeLength/2), 1.02*TMean],
        ['Thermal'  ,'1.04'     ,    0,  7,      1 ,    int(LatticeLength/2), 1.04*TMean],
        ['Hot'      ,'1.06'     ,    0,  8,      1 ,    int(LatticeLength/2), 1.06*TMean],
        ['Too hot'  ,'1.08'     ,    0,  9,      1 ,    int(LatticeLength/2), 1.08*TMean],
        ['Too hot'  ,'1.10'     ,    0, 10,      1 ,    int(LatticeLength/2), 1.10*TMean]
        ])
LPYD = LPYD[1:,:]
nz = np.nonzero(LPYD[:,2].astype(float))[0]
LPYD = LPYD[nz,:]



### Measurement Options ###
N_times = 10
N_tom = 10**20                    # Number of tomography experiments to run in each direction at each time point

### PCA Options ###
RunPCAonData = False
PCA_Var_Keep = 1
N_PCA_plot   = 1000                          # Number of data points to plot in PCA 


### tSNE Options ###
RunNNonData = False
f_train = 75
f_valid = 25
f_test = 0
nH1 = 20
nH2 = 'Skip'
dropout_prob = 0.5
L2reg = 0.01
learning_rate = 0.01
N_epochs = 100
minibatch_size = 10

fsum = f_train + f_valid + f_test
f_train = f_train/fsum
f_valid = f_valid/fsum
f_test  = f_test/fsum
