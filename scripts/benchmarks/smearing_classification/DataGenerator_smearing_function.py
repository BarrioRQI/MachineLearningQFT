#! /usr/bin/python
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import utils.utilities as u
import sys

############################################################################
##################### IMPORT AND SET PARAMETERS ############################
############################################################################

### Input parameters are set by DataGenerator_Parameters.py ###
import DataGenerator_Parameters_smearing_function as PDG
CaseString = PDG.CaseString	     # String associated with this run of the code
GenerateData = PDG.GenerateData  # Whether or not to generate new data
N_s = PDG.N_Samples              # Number of examples to produce for training/validating
Regression = PDG.Regression
save_data = PDG.save_data
save_intermediary_data = PDG.save_intermediary_data
verbose = PDG.verbose

### Quantum Field Parameters ###
LatticeLen = PDG.LatticeLen       # Length of Oscillator Lattice

### Probe/Coupling Parameters ###
sigma = PDG.sigma                # Probe Smearing Width
wD = PDG.wD                      # Probe Energy Gap   
lam = PDG.lam                    # Probe - Field Coupling Strength
mcc = PDG.m
eps = PDG.eps

### Measurement Options ###
Tmin = PDG.Tmin          # Defines measurement windows
Tmax = PDG.Tmax          # Defines measurement windows
Ntimes = PDG.N_times # Number of time steps considered in each scenario
MeasurementTimes = PDG.MeasurementTimes # Number of time steps considered in each scenario
Ntom = PDG.N_tom                  # Number of tomography experiments to run in each direction at each time point

### Defining Classes for Classification ###
T = PDG.T                # Mean Temperature for Thermal Cases

Cases   = PDG.Cases            # Number of cases being considered
YLabels = PDG.YLabels          # List of y-labels
plist   = PDG.plist            # List of probabilities
ylist   = PDG.ylist            # List of y-values
Blist   = PDG.Blist            # List of Boundary Conditions
Mlist   = PDG.Mlist            # List of field masses
Slist   = PDG.Slist            # List of smearings
Dlist   = PDG.Dlist
Smearinglist   = PDG.Smearinglist
CutoffList   = PDG.CutoffList

### PCA Options ###
RunPCAonData = PDG.RunPCAonData    # Whether or not to run PCA
PCA_Var_Keep = PDG.PCA_Var_Keep    # Fraction of variance to be kept
N_PCA_plot   = PDG.N_PCA_plot      # Number of data points to plot in PCA 

### Classification Options ###
RunNNonData = PDG.RunNNonData       # Whether or not to train the Neural Network
f_train = PDG.f_train               # Fraction of data used for training 
f_valid = PDG.f_valid               # Fraction of data used for validation
f_test = PDG.f_test                 # Fraction of data reserved for testing
nH1    = PDG.nH1                    # Number of neurons in the first hidden layer
L2reg = PDG.L2reg                   # L2 Regularizer
learning_rate = PDG.learning_rate   # Learning Rate
N_epochs = PDG.N_epochs             # Number of epoch to train over
minibatch_size = PDG.minibatch_size # Minibatch size

############################################################################
##################### INITIALIZATION #######################################
############################################################################

np.set_printoptions(precision=20, linewidth=1000)
mycwd = u.make_folder("..", CaseString)
output = np.zeros((2,len(MeasurementTimes)))

for t in range(len(MeasurementTimes)):
    #print(MeasurementTimes[t], N_times)
    if t == 0:
        continue
    Tmin = MeasurementTimes[t-1]
    Tmax = MeasurementTimes[t]
    mycwd = u.make_folder(mycwd, 'time_'+str(np.round(MeasurementTimes[t],3)) + '_Ntimes_'+str(Ntimes))

    ############################################################################
    ################# SIMULATE EXPERIMENTAL DATA ###############################
    ############################################################################


    if GenerateData == True: 
        u.make_folder(mycwd, 'ExpData_'+CaseString)
        ExpData0_master = np.zeros((Cases, 3*Ntimes+1))
        ExpData0 = np.zeros((Cases,N_s, 3*Ntimes+1))
        ExpData =  np.zeros((Cases*N_s, 3*Ntimes+1))

        for c in range(Cases):
            B = Blist[c]
            D = Dlist[c]
            sigma = Slist[c]
            smearing = Smearinglist[c]
            cutoff = CutoffList[c]

            aS = u.get_moment_step_switching(
                wD, mcc, lam, LatticeLen, sigma, B, D, T,
                Tmin, Tmax, Ntimes, Ntom, 
                smearing, cutoff, eps, TOM=False)
            
            ExpData0_master[c] = np.concatenate((aS,ylist[c]), axis=None)

            for s in range(N_s):
                aS_tom = u.Tomography(aS, N_tom=Ntom, med=np.zeros(aS.shape))
                ExpData[c*N_s+s] = np.concatenate((aS_tom,ylist[c]), axis=None) # Save this trajectory        
                ExpData0[c,s] = np.concatenate((aS_tom,ylist[c]), axis=None) # Save this trajectory        


        if save_intermediary_data:
            pd.DataFrame(ExpData0).to_csv('Raw_TrajData_All.csv',header=None,index=None)
        os.chdir("..")

        ############################################################################
        ################################ RUN PCA ON DATA ###########################
        ############################################################################

        if RunPCAonData == True:
            if GenerateData == False:
                OutputFile = CaseString+'\ExpData_'+CaseString
                ExpData0 = pd.read_csv(OutputFile+'\Raw_TrajData_All.csv',header=None).to_numpy()
            LO=1

            u.make_folder(mycwd, 'PCAdData_'+CaseString)
            PCAdData = u.run_PCA_on_data(ExpData, LO, f_train, Cases, PCA_Var_Keep)
            os.chdir("..")

        ############################################################################
        ############################ Training N.N. #################################
        ############################################################################

        if RunNNonData == True:
            if RunPCAonData == False:
                OutputFile = '\PCAdData_'+CaseString
                PCAdData = pd.read_csv(OutputFile+'\PCAd_CutTrajData_'+CaseString+'.csv',header=None).to_numpy()

            u.make_folder(mycwd, 'TrainingNN_'+CaseString)
            
            # define netowrk architecture
            nsamples = ExpData[:,:-LO].shape[0]
            train_valid_data = \
                u.get_valid_train_test_from_data(PCAdData, nsamples, f_train, f_valid, LO, Regression)

            nI = train_valid_data[2].shape[1]
            if Regression == False:
                nO = len(YLabels)
            else:
                nO = 1
            nn_data = u.define_NN(nI, nO, nH1, Regression, L2reg, learning_rate)

            if not Regression: minr, maxr, MDev = 0,0,0
            acc_valid = u.train_validate_NN(
                nn_data, 
                train_valid_data, 
                tuple([Regression, minr, maxr, MDev]), 
                tuple([nO, minibatch_size, Cases, N_epochs]), 
                verbose=100, 
                YLabels=YLabels)

        if save_data:
            output[0,t] = MeasurementTimes[t]
            output[1, 0] = Ntimes
            output[1, t] = np.mean(np.array(acc_valid[int(0.8*len(acc_valid)):]))
            os.chdir("..")
            os.chdir("..")
            mycwd = os.getcwd()
            np.save('BinaryStats', output)	
            pd.DataFrame(output).to_csv('BinaryStats.csv',header=None,index=None)