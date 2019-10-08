#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

### Meta Parameters ###
CaseString = 'Sign'
N_Samples    = 500                       # Number of scenarios to compute

### Gaussian option ###
QGaussian = True

### Probe Hamiltonian & Initial  State ###
wS                = 1                      # HS0 = wS Sz
InitialProbeState = 'Ground'               # Qubit: 'Ground' or 'Plus' or 'Excited'
                                           # Gauss: 'Ground'

### Environment Geometry and Hamiltonian###
QScalar = True
LatticeLength    = 10                       # Number of qubit systems along each axis
LatticeDimension = 1                        # Number of spatial dimensions in environment
wE       = 1                                # HE0 = wE (Sz*1*1 + 1*Sz*1 + 1*1*Sz + ...)
gEE      = 0.1                              # HE = HE0 + gEE sum_<ij> Hij
JEE_case = 'xx'                             # If 'XX'    , Hij = Sx Sx
                                            #    'ZZ'    , Hij = Sz Sz
                                            #    'SS'    , Hij = (1 1 + Sx Sx + Sy Sy + Sz Sz)/2
                                            #    'Random', Hij = Random Hamiltonian
                                            # If 'xx'    , Hij = x x
                                            #    'pp'    , Hij = p p
Gsignal = 3.1548                            # Squeezing of the "signal oscillator" 8db -> r = 1/2e^(0.8)
### Probe Coupling to Environment ###
gSA        = 0.1                            # Probe - Environment coupling strength
JSA_case   = 'xx'                           # If 'XX'    , HSA = Sx Sx
                                            #    'ZZ'    , HSA = Sz Sz
                                            #    'SS'    , HSA = (1 1 + Sx Sx + Sy Sy + Sz Sz)/2
                                            #    'Random', HSA = Random Hamiltonian
                                            # If 'xx'    , Hij = x x
                                            #    'pp'    , Hij = p p

### Parameters from quantum field
QuantumField = True

if QuantumField==True:

    sigma = 4.2                                            # Bohr radius in pm
    HbarCbySig = 7.14                                    # Hbar * c / Bohr radius in eV
    Ksig = 7                                              # Cutoff Factor
    a0 = np.pi/Ksig * sigma                               # Lattice a in pm
    L = 100.1*a0                                              # Distance to wall in pm
    mcc = 0.1                                               # Field mass in eV
    wD = 10                                               # Detector gap in eV

    
    E0 = HbarCbySig * Ksig/np.pi

    aUV = a0/a0
    sigmaSmear = sigma/a0                                                     
    LatticeLength = int(L/a0)
    wDet = wD/E0                                          
    mField = mcc/E0                                       
    lambdadet =1/2*np.sqrt((mField*wDet)/aUV)*wDet; #0.1 comes from wDet
    
    wS = wDet
    wE = mField
    gEE = 1/(mField*aUV**2)
    gSA = -2*lambdadet*np.sqrt(aUV/(mField*wDet));   # = wDet


### Defining the Dynamics ###
Tmin       = 0                              # Maximum evolution time
Tmax       = 25*3.14                        # Maximum evolution time
N_t        = 100 			                # Number of time steps in each scenario

### Picking Environment States ###
ThermalStateDist = 'Uniform'                 # If 'Gaussian', T = Gaussian(mu,stdev)
                                             #    'Uniform' , T = Uniform(mu.stdev)
ThermalStateMean = 32/127000                        # Mean Temperature for above distributions 
TMean= ThermalStateMean
ThermalStateDev = ThermalStateMean*1/100                      # Std. Dev for above distributions
TDev=ThermalStateDev

NonThermalStateDev = 0.1                     # Energy scale for non-thermality
                                             
LPYD1 = np.array([
        ['Case Name','Y Name'   ,'p_c','y','B.Case',     'Dist'         , 'Temp'],
        ['Uncut'    ,'Full Bond',    0,  0,      0 ,     LatticeLength-2,      0],
        ['Cut'      ,'No Bond'  ,    0,  1,      1 ,     LatticeLength-2,      0],
        ['Semi'     ,'Half Bond',    0,  2,      2 ,     LatticeLength-2,      0],
        ['Class'    ,'Class'    ,    0,  3,      3 ,     LatticeLength-2,      0],
        ['Signal'   ,'Signal'   ,    0,  4,      4 ,     LatticeLength-2,      0],
        ['Too cold' ,'0.9'      ,    1,  0,      0 ,     int(LatticeLength/2), 0.90*TMean],
        ['Cold'     ,'0.92'     ,    1,  1,      0 ,     int(LatticeLength/2), 0.92*TMean],
        ['Thermal'  ,'0.94'     ,    1,  2,      0 ,     int(LatticeLength/2), 0.94*TMean],
        ['Hot'      ,'0.96'     ,    1,  3,      0 ,     int(LatticeLength/2), 0.96*TMean],
        ['Too hot'  ,'0.98'     ,    1,  4,      0 ,     int(LatticeLength/2), 0.98*TMean],
        ['Too cold' ,'1.00'     ,    1,  5,      0 ,     int(LatticeLength/2), 1.00*TMean],
        ['Cold'     ,'1.02'     ,    1,  6,      0 ,     int(LatticeLength/2), 1.02*TMean],
        ['Thermal'  ,'1.04'     ,    1,  7,      0 ,     int(LatticeLength/2), 1.04*TMean],
        ['Hot'      ,'1.06'     ,    1,  8,      0 ,     int(LatticeLength/2), 1.06*TMean],
        ['Too hot'  ,'1.08'     ,    1,  9,      0 ,     int(LatticeLength/2), 1.08*TMean],
        ['Too hot'  ,'1.10'     ,    1, 10,      0 ,     int(LatticeLength/2), 1.10*TMean]
        ])
LPYD1 = LPYD1[1:,:]
nz = np.nonzero(LPYD1[:,2].astype(float))[0]
LPYD1 = LPYD1[nz,:]
    

## Thermometry calculation 
LPYD2 = np.array([
        ['Case Name','Y Name'   ,' p_c',' y',   'B.Case',          'Dist',         'TempMean' ],
        ['Too cold' ,'0.9'   ,     1,   0,         0 ,     int(LatticeLength/2), 0.90*TMean],
        ['Cold'     ,'0.92'  ,     1,   1,         0 ,     int(LatticeLength/2), 0.92*TMean],
        ['Thermal'  ,'0.94'  ,     1,   2,         0 ,     int(LatticeLength/2), 0.94*TMean],
        ['Hot'      ,'0.96'  ,     1,   3,         0 ,     int(LatticeLength/2), 0.96*TMean],
        ['Too hot'  ,'0.98'  ,     1,   4,         0 ,     int(LatticeLength/2), 0.98*TMean],
        ['Too cold' ,'1.00'  ,     1,   5,         0 ,     int(LatticeLength/2), TMean     ],
        ['Cold'     ,'1.02'  ,     1,   6,         0 ,     int(LatticeLength/2), 1.02*TMean],
        ['Thermal'  ,'1.04'  ,     1,   7,         0 ,     int(LatticeLength/2), 1.04*TMean],
        ['Hot'      ,'1.06'  ,     1,   8,         0 ,     int(LatticeLength/2), 1.06*TMean],
        ['Too hot'  ,'1.08'  ,     1,   9,         0 ,     int(LatticeLength/2), 1.08*TMean],
        ['Too hot'  ,'1.10'  ,     1,   10,        0 ,     int(LatticeLength/2), 1.10*TMean]
        ])
    
  
LPYD2 = LPYD2[1:,:]
nz = np.nonzero(LPYD2[:,2].astype(float))[0]
LPYD2 = LPYD2[nz,:]



### Measurement Options ###
N_tom = 10**10                    # Number of tomography experiments to run in each direction at each time point

### PCA Options ###
RunPCAonData = True
PCA_Var_Keep = 1
N_PCA_plot   = 1000                          # Number of data points to plot in PCA 


### tSNE Options ###
RuntSNEonData = False
N_plot_tSNE   = 500                         # Number of data points to plot in tSNE
LeaveOff_tSNE = 2                            # Number of columns to neglect in tSNE

### tSNE Options ###
RunNNonData = True
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
