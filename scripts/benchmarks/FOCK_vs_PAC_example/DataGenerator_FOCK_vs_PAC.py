#! /usr/bin/python
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import pandas as pd
#import pylab as plt
from Code.Utilities import generate_moment, get_quadrature
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import Utilities as u
import sys

############################################################################
##################### IMPORT AND SET PARAMETERS ############################
############################################################################

### Input parameters are set by DataGenerator_Parameters.py ###
import DataGenerator_Parameters_FOCK_VS_PAC as PDG
CaseString = PDG.CaseString	     # String associated with this run of the code
GenerateData = PDG.GenerateData  # Whether or not to generate new data
N_s = PDG.N_Samples              # Number of examples to produce for training/validating
Regression = PDG.Regression

### Probe/Coupling Parameters ###
sigma = PDG.sigma                # Probe Smearing Width
L = PDG.L                      # Probe Energy Gap   
wD_list = PDG.wD_list                      # Probe Energy Gap   

Tmin = PDG.Tmin
Tmax = PDG.Tmax
N_cutoff = PDG.N_cutoff
N = PDG.N

### Measurement Options ###
N_times    = PDG.N_times		 # Number of time steps considered in each scenario
N_tom = PDG.N_tom                # Number of tomography experiments to run in each direction at each time point

### Defining Classes for Classification ###
Cases   = PDG.Cases            # Number of cases being considered
YLabels = PDG.YLabels          # List of y-labels
plist   = PDG.plist            # List of probabilities
ylist   = PDG.ylist            # List of y-values

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



# hamiltonian represented by symplectic form F, where F is in basis (qD, pD, q1, p1, ..., qn, pn)

output = np.zeros((7,len(wD_list)))
for z, wD in enumerate(wD_list):
    lam0 = 0.01

    os.chdir(BaseFile)
    CaseString = 'Freq'+str(np.round(wD,3))        
    CaseFile = CaseString
    try:
        os.mkdir(CaseFile)
    except FileExistsError:
        pass
    os.chdir(BaseFile + "/" + CaseFile)
    mycwd = os.getcwd()
    print('Creating Directory:',CaseString)


    projectors = list()
    for a, b in zip([1, 0, 1/np.sqrt(2)], [0, 1, 1/np.sqrt(2)]):
        projectors.append(get_quadrature(a, b, N_cutoff))

    tList = np.linspace(Tmin, Tmax, N_times+1)[1:]
    print('Simulating trajectories ...')
    # populate experimental data
    moments = np.zeros((Cases, 6*N_times))
    moments_p = np.zeros((Cases, 6*N_times))
    np.set_printoptions(precision=10, linewidth=1000)

    for c in range(Cases):
        for i, t in enumerate(tList):
            for j, M in enumerate([2, 4]):
                for k, p in enumerate(projectors):
                    m = generate_moment(N, M, YLabels[c], p, 0, t, lam0, N_cutoff, sigma, L, wD)
                    m_p = generate_moment(N, M*2, YLabels[c], p, 0, t, lam0, N_cutoff, sigma, L, wD)
                    moments[c, i*6 + 3*j + k] = m
                    moments_p[c, i*6 + 3*j + k] = m_p

    ############################################################################
    ################# SIMULATE EXPERIMENTAL DATA ###############################
    ############################################################################
    print('Simulating experimental data ...')

    if GenerateData == True: 
        os.chdir(mycwd)
        OutputFile = 'ExpData_'+CaseString
        try:
            os.mkdir(OutputFile)
        except FileExistsError:
            pass
        os.chdir(mycwd + "/" + OutputFile)

        ExpData0 = np.zeros((Cases, N_s, 6*N_times+1))
        ExpData = np.zeros((Cases*N_s, 6*N_times+1))
         
        for c in range(Cases):
            extrainfo = np.array([ylist[c]])                           # Define extra information about this case
            LO = len(extrainfo)                                        # Define the "leave off" length

            for s in range(N_s):                                    
                
                moments_tom = u.moment_Tomography(moments[c], moments_p[c], N_tom=N_tom)                            # Add tomographic noise
                ExpData0[c,s] = np.concatenate((moments_tom,extrainfo), axis=None)
                ExpData[c*N_s+s] = np.concatenate((moments_tom,extrainfo), axis=None) # Save this trajectory        
                
                if N_s < 20:
                    pass
                elif s % int(N_s/20) == 0:
                    sys.stdout.write('\r')
                    sys.stdout.write('Data points created for case %d of %d: %d Percent complete: %d' % (c+1, Cases+1, s, int(100*s/N_s)))
            
            #pd.DataFrame(ExpData0[c]).to_csv('Raw_TrajData_'+YLabels[c]+'.csv',header=None,index=None)
        print()

    ############################################################################
    ################################ RUN PCA ON DATA ###########################
    ############################################################################
    print('Running PCA ...')

    if RunPCAonData == True:
        if GenerateData == False:
            OutputFile = CaseFile+'\ExpData_'+CaseString
            ExpData = pd.read_csv(OutputFile+'\Raw_TrajData_All.csv',header=None).to_numpy()
            LO=1

        os.chdir(mycwd)
        OutputFile = 'PCAdData_'+CaseString
        try:
            os.mkdir(OutputFile)
        except FileExistsError:
            pass        

        os.chdir(mycwd + "/"  + OutputFile)

        ### Randomly shuffle the indices ###
        permut = np.arange(ExpData.shape[0])
        np.random.shuffle(permut) 
        ExpData = ExpData[permut,:]
        X = ExpData[:,:-LO]
        y = ExpData[:,-1:]

    
        n_train = int(f_train*X.shape[0])  # Number of data points used for training
        n_valid = int(f_valid*X.shape[0])  # Number of data points used for validating     
    
        X_train = X[:n_train]            # X is the unlabeled data points 
        y_train = y[:n_train]            # Y is the labels 

        Xm, lam, M = u.PCA(X_train)                          # Run PCA on the data
        
        X = np.dot(X - np.tile(Xm, (X.shape[0], 1)), M.T).real
        X = X/np.sqrt(lam.T)
        PCAdData = np.append(X,y.reshape((len(y),1)),axis=1) # Add labels back to PCA'd data
    
        pd.DataFrame(Xm).to_csv('PCA_Mean_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(lam).to_csv('PCA_EigVal_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(M).to_csv('PCA_EigVec_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(PCAdData).to_csv('PCAd_TrajData_'+CaseString+'.csv',header=None,index=None)


        d_0 = len(lam)
        if PCA_Var_Keep == 'All':
            d_c = d_0
        elif 0 < PCA_Var_Keep < 1:
            d_c = u.PCA_Compress(lam,PCA_Var_Keep)
        else:
            d_c = u.PCA_Compress(lam,1-10**(-10))
            
        d_c = max(d_c,Cases,3)
        M = M[:d_c,:]
        dlist = list(range(PCAdData.shape[1]))
        del dlist[d_c:-1]
        PCAdData = PCAdData[:,dlist]
        
        pd.DataFrame(M).to_csv('PCA_CutEigVec_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(PCAdData).to_csv('PCAd_CutTrajData_'+CaseString+'.csv',header=None,index=None)
    
        os.chdir("..")

    ############################################################################
    ##################### PROCESSING DATA FOR N.N. #############################
    ############################################################################

    print('Training NN ...')

    if RunNNonData == True:
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
        
        
        ############################################################################
        ##################### DEFINE THE NETWORK ARCHITECTURE ######################
        ############################################################################
        
        if Regression == True:
            y_train = np.reshape(y_train,(y_train.shape[0],1))
            y_valid = np.reshape(y_valid,(y_valid.shape[0],1))
            y_test = np.reshape(y_test,(y_test.shape[0],1))
        
        nI = x_train.shape[1]
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

                if epoch % 10 == 0:
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

    output[0,z] = wD
    output[1,z] = np.mean(np.array(acc_valid[int(0.8*len(acc_valid)):]))
    output[2,z] = np.std(np.array(acc_valid[int(0.8*len(acc_valid)):]))
    
    os.chdir("..")
    os.chdir("..")

    print(mycwd)

    np.save('BinaryStats', output)	
    pd.DataFrame(output).to_csv('BinaryStats.csv',header=None,index=None)