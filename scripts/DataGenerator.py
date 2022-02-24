#! /usr/bin/python
import os
import numpy as np
import pandas as pd
import pylab as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import Utilities as u
import sys

############################################################################
##################### IMPORT AND SET PARAMETERS ############################
############################################################################

### Input parameters are set by DataGenerator_Parameters.py ###
import DataGenerator_Parameters as PDG
CaseString = PDG.CaseString	     # String associated with this run of the code
GenerateData = PDG.GenerateData  # Whether or not to generate new data
N_s = PDG.N_Samples              # Number of examples to produce for training/validating
Regression = PDG.Regression

### Quantum Field Parameters ###
LatLen = PDG.LatticeLength       # Length of Oscillator Lattice
mcc = PDG.mcc                    # Field Mass

### Probe/Coupling Parameters ###
sigma = PDG.sigma                # Probe Smearing Width
wD = PDG.wD                      # Probe Energy Gap   
lam = PDG.lam                    # Probe - Field Coupling Strength

### Measurement Options ###
PlotTimes = PDG.PlotTimes        # Defines measurement windows
N_times    = PDG.N_times		 # Number of time steps considered in each scenario
N_tom = PDG.N_tom                # Number of tomography experiments to run in each direction at each time point

### Defining Classes for Classification ###
TMean = PDG.TMean                # Mean Temperature for Thermal Cases
TDev  = PDG.TDev                 # Size of Temperature Bins
Gsignal= PDG.Gsignal             # Squeezing of the "signal oscillator"

Cases   = PDG.Cases            # Number of cases being considered
YLabels = PDG.YLabels          # List of y-labels
plist   = PDG.plist            # List of probabilities
ylist   = PDG.ylist            # List of y-values
Blist   = PDG.Blist            # List of Boundary Conditions
Dlist   = PDG.Dlist            # List of Distances to Boundary
TempList = PDG.TempList        # List of Temperatures

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
#print(' ')
#print(' ')

os.chdir("..")
mycwd = os.getcwd()

CaseFile = CaseString
try:
    os.mkdir(CaseFile)
except FileExistsError:
    pass
print('Created Directory:',CaseFile) 
os.chdir(mycwd + "/" + CaseFile)
mycwd = os.getcwd()
BaseFile = os.getcwd()


#print('Computing Hamiltonians')
Hlist_dynamic = [0]*Cases         # List of Hamiltonians for dynamics, for each scenario
Hlist_thermal = [0]*Cases         # List of Hamiltonians for thermality
for k in range(Cases):
    B = Blist[k]
    D = Dlist[k]
    Hlist_dynamic[k], Hlist_thermal[k] = u.ComputeHams(wD,mcc,lam,LatLen,sigma,B,D)
    #print(Hlist_dynamic[k])
#print('Done!')

# hamiltonian represented by symplectic form F, where F is in basis (qD, pD, q1, p1, ..., qn, pn)

#print('Considering times:',PlotTimes)
output = np.zeros((7,len(PlotTimes)))
for t in range(len(PlotTimes)):
    if t == 0:
        continue
    #print(' ')
    #print(' ')
    os.chdir(BaseFile)
    CaseString = 'Time'+str(np.round(PlotTimes[t],3))        
    CaseFile = CaseString
    try:
        os.mkdir(CaseFile)
    except FileExistsError:
        pass
    os.chdir(BaseFile + "/" + CaseFile)
    mycwd = os.getcwd()
    print('Creating Directory:',CaseString)

    Tmax = PlotTimes[t]
    Tmin = PlotTimes[t-1]    

    #print('Computing Unitaries and Evolved Projectors')
    ProjList = [0]*Cases
    for k in range(Cases):
        ProjList[k] = u.DefProjList(Hlist_dynamic[k],N_times,Tmin,Tmax)
    #print('Done!')
    #print(ProjList)
    MedProj = np.median(ProjList,axis=0) # Compute the median projector
    #print(MedProj)
    ############################################################################
    ######################## COMPUTING EXACT STATISTICS ########################
    ############################################################################
    np.set_printoptions(precision=20, linewidth=1000)
    #print(ProjList)
    #print('Precomputing Exact Trajectories')
    OutputFile = 'ExactSolutions_'+CaseString
    os.chdir(mycwd)
    try:
        os.mkdir(mycwd + '/' + OutputFile)
    except FileExistsError:
        pass
    
    os.chdir(mycwd + "/" + OutputFile)

    RS0 = u.InitializeProbeState('Ground')
    #print(RS0)
    RSE0List = [0]*Cases                                   # List of probe-environment initial states
    for k in range(Cases):
        #print('case')
        RE0 = u.ThermalState(Hlist_thermal[k],TempList[k]) # Compute the environment's thermal state
        #print(RE0[0, 0])
        #print(TempList[k])

        if Blist[k] == 3:                                         # In the signaling case
            RE0[0,0]= Gsignal                              # Squeeze the last oscillator
            RE0[0,LatLen]= 0                               
            RE0[LatLen,0]= 0
            RE0[LatLen,LatLen]= 1/Gsignal    
        RSE0List[k] = u.directsum(RE0,RS0)                 # Compute the initial probe-environment state
    #print('FULL LIST')
    #print(RSE0List)
    ##print(np.around(np.asarray(RSE0List), 3))
    MedRSE0 = np.median(RSE0List,axis=0)                   # Compute the median probe-environment stat
    ##print(MedRSE0)

    ### Computing "Median" Probe Trajectory ###
    d1 = MedProj.shape[0]
    d2 = MedProj.shape[1]
    MedAS = np.zeros((d1,d2))
    for n in range(d1):                                            
        for r in range(d2):
             MedAS[n,r] = np.trace(MedProj[n,r] @ MedRSE0).real
    ##print(MedAS)
    MedAS = np.array(MedAS).flatten().real
    #print(MedAS)
    #print(MedRSE0)
    #print(MedProj.shape)
    #print(len(ProjList), ProjList[0].shape)

    ### Calculating probe trajectory in for each case
    PrePickedAS = [0]*(Cases+1)
    PrePickedAS[Cases] = MedAS
    #print('CASES')
    for k in range(Cases):        
        dP = ProjList[k] - MedProj      # Difference from median projector
        dS = RSE0List[k] - MedRSE0      # Difference from median state
        #assert(np.count_nonzero(dS) > 0)
        d1 = ProjList[k].shape[0]
        d2 = ProjList[k].shape[1]
        aS = np.zeros((d1,d2))
        for n in range(d1):             # Compute difference from "median" trajectory
            for r in range(d2):
                 aS[n,r] += np.trace(dP[n,r] @ MedRSE0).real
                 aS[n,r] += np.trace(MedProj[n,r] @ dS).real
                 aS[n,r] += np.trace(dP[n,r] @ dS).real
        #print(aS, k, Cases)
        #print()
        #assert(np.count_nonzero(aS) > 0)
        PrePickedAS[k] = np.array(aS).flatten().real
        #print('Done for case',k+1,'of',Cases,'!')
    #print(aS)
    #print(aS.shape)
    #assert(np.count_nonzero(aS) > 0)
    ### Picking Random initial states for thermal case ###
    if TDev != 0 and t == 1:
        if Regression == True: reglist = np.zeros((Cases*N_s,))
        d1=RSE0List[0].shape[0]
        d2=RSE0List[0].shape[1]
        BigDSList = np.zeros((Cases*N_s,d1,d2))
        for c in range(Cases):
            #print('Creating states for case',c+1,' of ',Cases,'!')
            for s in range(N_s):
                Temp = u.RTemp(TempList[c],TDev)
                RE0 = u.ThermalState(Hlist_thermal[c],Temp)
                RSE0 = u.directsum(RE0,RS0)
                BigDSList[c*N_s+s] = RSE0 - MedRSE0
                if Regression == True: reglist[c*N_s+s] = Temp
                    
                if N_s < 20:
                    pass
                elif s % int(N_s/20) == 0:
                    sys.stdout.write('\r')
                    sys.stdout.write('States created for case %d of %d: %d Percent complete: %d' % (c+1, Cases+1, s, int(100*s/N_s)))
    print()
    #print('Outputing Exact Trajectory Data to csv')
    #b2.save(OutputFile+'\ExactBlochTraj_All.pdf')
    pd.DataFrame(PrePickedAS).to_csv('ExactTrajData_'+CaseString+'.csv',header=None,index=None)
    #print('Done!')
    
    if Regression == True and t == 1:
        minr = min(reglist)
        maxr = max(reglist)
        reglist = reglist - minr
        reglist = reglist/(maxr-minr)
        reglist = 0.5*reglist
        reglist = reglist + 0.25

    

    ############################################################################
    ##################### HELLINGER DISTANCE & log(N(1/2)) #####################
    ############################################################################
    if TDev != 0:
        print('Cannot Compute Hellinger Distance when TDev != 0')
    else:
        #print('Computing Hellinger Distance')
        Hell = np.zeros((Cases,Cases))
        Nhalf = np.zeros((Cases,Cases))

        MedAS = PrePickedAS[Cases]            
        for c1 in range(Cases):
            for c2 in range(Cases):
                dmu = PrePickedAS[c1] - PrePickedAS[c2]
                sig1 = 2*(MedAS + PrePickedAS[c1])**2
                sig2 =  2*(MedAS + PrePickedAS[c2])**2                
                Hellinger, N_half = u.ComputeHellinger(dmu,sig1,sig2,N_tom-1)
                                
                Hell[c1,c2] = Hellinger         # Hellinger Distance between cases c1 and c2
                Nhalf[c1,c2] = N_half           # Amount of tomography to get H**2 = 0.5
            DF_Hell = pd.DataFrame.from_dict(dict([(YLabels[k],Hell[k]) for k in range(len(YLabels))]),orient='index', columns=YLabels)
            DF_Hell.to_csv('Hell_'+CaseString+'.csv')
    
    
    ############################################################################
    ################# SIMULATE EXPERIMENTAL DATA ###############################
    ############################################################################
    if GenerateData == True: 
        #print('Creating data')
        os.chdir(mycwd)
        OutputFile = 'ExpData_'+CaseString
        try:
            os.mkdir(OutputFile)
        except FileExistsError:
            pass
	
        os.chdir(mycwd + "/" + OutputFile)
            
        ExpData0 = np.zeros((Cases,N_s,3*N_times+1))
        ExpData = np.zeros((Cases*N_s,3*N_times+1))
         
        MedAS = PrePickedAS[Cases]
        for c in range(Cases):
            #print('Creating data for case',c+1,' of ',Cases,'!')
            aS = PrePickedAS[c]                                        # Look up the exact trajectory for this case
            extrainfo = np.array([ylist[c]])                           # Define extra information about this case
            LO = len(extrainfo)                                        # Define the "leave off" length

            # In thermal case compute environment state independent part of the probe trajectory
            if TDev != 0:                                              
                dP = ProjList[c] - MedProj
                d1 = ProjList[c].shape[0]
                d2 = ProjList[c].shape[1]
                aS1 = np.zeros((d1,d2))
                for n in range(d1):
                    for r in range(d2):
                        aS1[n,r] += np.trace(dP[n,r] @ MedRSE0).real

            for s in range(N_s):                
                # In thermal case compute environment state dependent part of the probe trajectory
                if TDev != 0:                
                    dS = BigDSList[c*N_s+s]
                    d1 = ProjList[c].shape[0]
                    d2 = ProjList[c].shape[1]
                    aS = np.zeros((d1,d2))
                    for n in range(d1):
                        for r in range(d2):
                            aS[n,r] += aS1[n,r]
                            aS[n,r] += np.trace(ProjList[c][n,r] @ dS).real                         
                    aS = np.array(aS).flatten().real
                    if Regression == True: extrainfo = np.array([reglist[c*N_s+s]])
                    
                aS_tom = u.Tomography(aS,N_tom,MedAS)                            # Add tomographic noise
                print(extrainfo)
                print(len(aS_tom))
                ExpData0[c,s] = np.concatenate((aS_tom,extrainfo), axis=None)
                ExpData[c*N_s+s] = np.concatenate((aS_tom,extrainfo), axis=None) # Save this trajectory        
                if N_s < 20:
                    pass
                elif s % int(N_s/20) == 0:
                    sys.stdout.write('\r')
                    sys.stdout.write('Data points created for case %d of %d: %d Percent complete: %d' % (c+1, Cases+1, s, int(100*s/N_s)))
            #print('Data points created:',N_s ,'Percent complete:', 100 )
            pd.DataFrame(ExpData0[c]).to_csv('Raw_TrajData_'+YLabels[c]+'.csv',header=None,index=None)
        print()

        #print('Outputing Raw Trajectory Data to csv')
        pd.DataFrame(ExpData).to_csv('Raw_TrajData_All.csv',header=None,index=None)
        #print('Done!')
                
        os.chdir("..")
        
        #print(mycwd)
    print(ExpData)
    print(ExpData.shape)
    ############################################################################
    ################################ RUN PCA ON DATA ###########################
    ############################################################################
    if RunPCAonData == True:
        if GenerateData == False:
            OutputFile = CaseFile+'\ExpData_'+CaseString
            #print('Loading in Data')
            ExpData = pd.read_csv(OutputFile+'\Raw_TrajData_All.csv',header=None).to_numpy()
            LO=1
            #print('Loaded Data')

        os.chdir(mycwd)
        OutputFile = 'PCAdData_'+CaseString
        try:
            os.mkdir(OutputFile)
        except FileExistsError:
            pass        

        os.chdir(mycwd + "/"  + OutputFile)

        ### Randomly shuffle the indices ###
        #print('Shuffling Data')
        permut = np.arange(ExpData.shape[0])
        np.random.shuffle(permut) 
        ExpData = ExpData[permut,:]
        X = ExpData[:,:-LO]
        y = ExpData[:,-1:]

    
        n_train = int(f_train*X.shape[0])  # Number of data points used for training
        n_valid = int(f_valid*X.shape[0])  # Number of data points used for validating     
    
        X_train = X[:n_train]            # X is the unlabeled data points 
        y_train = y[:n_train]            # Y is the labels 

        #print('Running PCA on Training Data')    
        Xm, lam, M = u.PCA(X_train)                          # Run PCA on the data
        
        X = np.dot(X - np.tile(Xm, (X.shape[0], 1)), M.T).real
        X = X/np.sqrt(lam.T)
        PCAdData = np.append(X,y.reshape((len(y),1)),axis=1) # Add labels back to PCA'd data
    
        #print('Outputing Eigensystem and PCA\'d Data to csv')
        pd.DataFrame(Xm).to_csv('PCA_Mean_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(lam).to_csv('PCA_EigVal_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(M).to_csv('PCA_EigVec_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(PCAdData).to_csv('PCAd_TrajData_'+CaseString+'.csv',header=None,index=None)
        #print('Done!')


        d_0 = len(lam)
        if PCA_Var_Keep == 'All':
            d_c = d_0
        elif 0 < PCA_Var_Keep < 1:
            #print('Compressing Data with PCA')    
            d_c = u.PCA_Compress(lam,PCA_Var_Keep)
            #print('Input dimension compressed to',d_c,'of',d_0,'dimensions, keeping', 100*PCA_Var_Keep,'percent of the variance.')
        else:
            #print('Removing extremely low variance dimensions')
            d_c = u.PCA_Compress(lam,1-10**(-10))
            #print(d_0 - d_c,'of',d_0,'dimensions had extremely low variance and were removed.')
            
        d_c = max(d_c,Cases,3)
        M = M[:d_c,:]
        dlist = list(range(PCAdData.shape[1]))
        del dlist[d_c:-1]
        PCAdData = PCAdData[:,dlist]
        
        pd.DataFrame(M).to_csv('PCA_CutEigVec_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(PCAdData).to_csv('PCAd_CutTrajData_'+CaseString+'.csv',header=None,index=None)
        '''
        if Regression == False: 
            #print('Plotting Data in Max Variance Plane')
            Var2_state = 100*(lam[0]+lam[1])/sum(lam)
            ymin = np.min(y)
            ymax = np.max(y)
            plt.close()
            plt.title('PCA\'d Data: Varariance Covered %1.3f' % Var2_state)
            plt.axes().set_aspect('equal', 'datalim')
            cm = plt.get_cmap('tab10', ymax-ymin+1)
            scat = plt.scatter(PCAdData[:N_PCA_plot,0], PCAdData[:N_PCA_plot,1], c = PCAdData[:N_PCA_plot,-1], cmap=cm,vmin = ymin-.5, vmax = ymax+.5)
            cb = plt.colorbar(scat, ticks=np.arange(ymin,ymax+1))
            cb.ax.set_yticklabels(YLabels)
            plt.savefig('Fig_PCA_'+CaseString+'.pdf')
            #print('Done!')
        '''
        os.chdir("..")
        #print(mycwd)
    
    ############################################################################
    ##################### PROCESSING DATA FOR N.N. #############################
    ############################################################################

    if RunNNonData == True:
        #print('Processing Data for Neural Network Training')
        if RunPCAonData == False:
            OutputFile = '\PCAdData_'+CaseString
            PCAdData = pd.read_csv(OutputFile+'\PCAd_CutTrajData_'+CaseString+'.csv',header=None).to_numpy()

        OutputFile = 'TrainingNN_'+CaseString
        os.chdir(mycwd)
        try:
            os.mkdir(OutputFile)
        except FileExistsError:
            pass  
  
        os.chdir(mycwd + "/"  + OutputFile)
    
        n_train = int(f_train*X.shape[0])  # Number of data points used for training
        n_valid = int(f_valid*X.shape[0])  # Number of data points used for validating     
        x_train = PCAdData[:n_train,:-LO]
        x_valid = PCAdData[n_train:n_train+n_valid,:-LO]
        x_test  = PCAdData[n_train+n_valid:,:-LO]
        y_train = PCAdData[:n_train,-1:].flatten()
        y_valid = PCAdData[n_train:n_train+n_valid,-1:].flatten()
        y_test  = PCAdData[n_train+n_valid:,-1:].flatten()    
        
        #print('Done!')
        
        ############################################################################
        ##################### DEFINE THE NETWORK ARCHITECTURE ######################
        ############################################################################
        
        #print('Defining Network Architecture')
        if Regression == True:
            y_train = np.reshape(y_train,(y_train.shape[0],1))
            y_valid = np.reshape(y_valid,(y_valid.shape[0],1))
            y_test = np.reshape(y_test,(y_test.shape[0],1))
        
        nI = x_train.shape[1]
        #print(YLabels)
        if Regression == False:
            nO = len(YLabels)
        else:
            nO = 1

        X = tf.placeholder("float", [None, nI])
        if Regression == True:
            Y = tf.placeholder("float", [None, nO])
        else:
            Y = tf.placeholder(tf.int32,[None])
                   
        weights = {
                'h1': tf.Variable(tf.random_normal([nI, nH1],mean=0.0,stddev=np.sqrt(2/(nI+nH1)))),
                'out': tf.Variable(tf.random_normal([nH1, nO],mean=0.0,stddev=np.sqrt(2/(nH1+nO)))) 
                }
        biases = {
                'b1': tf.Variable(tf.zeros([nH1])),
                'out': tf.Variable(tf.zeros([nO]))
                }           
                
        def neural_net(x,QReg=False):
            #hidden layer 1
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            # output layer
            out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
            if QReg == False: out_layer = tf.nn.softmax(out_layer)
            if QReg == True: out_layer = tf.sigmoid(out_layer)
            return out_layer

        Y_hat=neural_net(X,QReg = Regression)            

        L2cost = L2reg*(tf.nn.l2_loss(weights['h1'])+tf.nn.l2_loss(weights['out']))
        if Regression == False:
            Y_onehot = tf.one_hot(Y,depth = nO)
            eps = 10**(-10) # to prevent the logs from diverging
            cross_entropy = tf.reduce_mean(-tf.reduce_sum( Y_onehot * tf.log(Y_hat+eps), reduction_indices=[1]))
            loss_op = cross_entropy + L2cost
        if Regression == True:
            mse_cost = tf.losses.mean_squared_error(Y,Y_hat)
            loss_op = mse_cost + L2cost

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)#define optimizer # play around with learning rate
        train_op = optimizer.minimize(loss_op)#minimize losss

        epoch_list    = [-1]
        cost_training = [np.nan]
        cost_valid    = [np.nan]
        acc_training  = [1/Cases]
        acc_valid     = [1/Cases]
        if Regression == True:
            acc_training  = [np.nan]
            acc_valid     = [np.nan]
        disp_list =[]

        #print("Beginning Training")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())                
            
            permut = np.arange(n_train)
            for epoch in range(N_epochs+1):
                np.random.shuffle(permut) # Randomly shuffle the indices
                x_shuffled = x_train[permut,:]
                y_shuffled = y_train[permut]
                for b in range(0, n_train, minibatch_size):                    #Loop over all the mini-batches:
                    x_batch = x_shuffled[b:b+minibatch_size,:]
                    y_batch = y_shuffled[b:b+minibatch_size]
                    sess.run(train_op,feed_dict={X:x_batch,Y:y_batch})
                
                if epoch % 1 == 0:
                    cost_tr=sess.run(loss_op,feed_dict={X:x_train,Y:y_train})
                    cost_va=sess.run(loss_op,feed_dict={X:x_valid,Y:y_valid})
                    NN_output_tr=sess.run(Y_hat,feed_dict={X:x_train})
                    NN_output_va=sess.run(Y_hat,feed_dict={X:x_valid})                    
                    if Regression == True:
                        temphat_tr = (NN_output_tr-1/4)*2*(maxr-minr)+minr
                        temphat_va = (NN_output_va-1/4)*2*(maxr-minr)+minr
                        temp_tr = (y_train-1/4)*2*(maxr-minr)+minr
                        temp_va = (y_valid-1/4)*2*(maxr-minr)+minr
                        acc_tr = np.mean(abs(temphat_tr - temp_tr) < 0.01*temp_tr)
                        acc_va = np.mean(abs(temphat_va - temp_va) < 0.01*temp_va)
                    else:
                        predicted_class_tr = np.argmax(NN_output_tr, axis=1)
                        acc_tr = np.mean(predicted_class_tr == y_train)
                        predicted_class_va = np.argmax(NN_output_va, axis=1)
                        acc_va = np.mean(predicted_class_va == y_valid)

                    if Regression == False:
                        valid_matrix = np.zeros((nO,nO))
                        counter = np.zeros((nO,))
                        for k in range(len(y_valid)):
                            valid_matrix[int(predicted_class_va[k]),int(y_valid[k])] += 1
                            counter[int(y_valid[k])] += 1
                        for s in range(nO): 
                            valid_matrix[:,s] = valid_matrix[:,s]/counter[s]
                        g = np.zeros((nO,1))
                        for s in range(nO):
                            gs = sum(valid_matrix[s,:])
                            if gs == 0:
                                g[s] == 1/Cases
                            else:
                                g[s] = valid_matrix[s,s]/gs
                        valid_matrix = np.append(valid_matrix,g,axis=1)
                    
                    epoch_list.append(epoch)
                    cost_training.append(cost_tr)
                    cost_valid.append(cost_va)
                    acc_training.append(acc_tr)
                    acc_valid.append(acc_va)
                    if Regression == False: disp_list.append(valid_matrix)

                if epoch % 100 == 0:
                    print( "Iteration %d:\n  Training cost %f \n  Validation cost %f\n  Training accuracy %f\n  Validating accuracy %f\n" % (epoch, cost_tr, cost_va, acc_tr, acc_va) )
                    if Regression == False:
                        Display = pd.DataFrame.from_dict(dict([(YLabels[k],valid_matrix[k]) for k in range(len(YLabels))]),orient='index', columns=YLabels + ['Pres.'])
                        print(Display)
                    print(' ')
                    print(' ')
        '''
        fig = plt.figure(1,figsize=(10,5))
        fig.subplots_adjust(hspace=.3,wspace=.3)
        plt.clf()
        ### Plot the training accuracy: ###
        plt.subplot(211)
        plt.plot(epoch_list,acc_training,'b-',epoch_list,acc_valid,'r-')
        plt.xlabel('Epoch')
        plt.ylabel('Acc.')
        # plt.yscale('log')
        
        ### Plot the cost function during training: ###
        plt.subplot(212)
        plt.plot(epoch_list,cost_training,'b-',cost_valid,'r-')
        plt.xlabel('Epoch')
        plt.ylabel('Cost Func')
        # plt.yscale('log')
            
        plt.savefig('Fig_Class_'+CaseString+'.pdf')
        '''
            
            
    if TDev == 0:

        output[0,t] = PlotTimes[t]
        output[1,t] = np.mean(np.array(acc_valid[int(0.8*len(acc_valid)):]))
        output[2,t] = np.std(np.array(acc_valid[int(0.8*len(acc_valid)):]))
        #output[2,t] = TV[0,1]
        output[3,t] = Hell[0,1]
        output[4,t] = Nhalf[0,1]
        output[5,t] = Hell[1,0]
        output[6,t] = Nhalf[1,0]
        
        os.chdir("..")
        os.chdir("..")

        print(mycwd)

        np.save('BinaryStats', output)	
        pd.DataFrame(output).to_csv('BinaryStats.csv',header=None,index=None)
    
    if TDev != 0:
        #prec1=np.array(disp_list)[:,0,-1]
        #prec2=np.array(disp_list)[:,1,-1]
        #prec3=np.array(disp_list)[:,2,-1]
        output[0,t] = PlotTimes[t]
        #output[1,t] = np.mean(np.array(prec1[int(0.8*len(prec1)):]))
        #output[2,t] = np.mean(np.array(prec2[int(0.8*len(prec2)):]))
        #output[3,t] = np.mean(np.array(prec3[int(0.8*len(prec3)):]))
        output[4,t] = np.mean(np.array(acc_valid[int(0.8*len(acc_valid)):]))

        os.chdir("..")
        os.chdir("..")

        np.save('BinaryStats', output)
        pd.DataFrame(output).to_csv('BinaryStats.csv',header=None,index=None)