import os
import numpy as np
import pandas as pd
import pylab as plt
import tensorflow as tf
import Utilities as u

############################################################################
##################### IMPORT AND SET META PARAMETERS #######################
############################################################################

### Input parameters are set by DataGenerator_Parameters.py ###
import DataGenerator_Parameters as PDG

GenerateData = PDG.GenerateData

CaseString = PDG.CaseString		# The string which labels this run of the code
LPYD = PDG.LPYD

# Identifying the Y labels 
DupYLabels = LPYD[:,1]          # Extract the y labels from LPYD (with duplicates)
YLabels = []                    # Remove the duplicates
for i in DupYLabels: 
    if i not in YLabels: 
        YLabels.append(i)

# Normalize the relative probabilities of each case
plist = LPYD[:,2].astype(float) # Extract probabilities from LYPD
ptot  = sum(plist)              # Compute Total
plist = plist/ptot              # Normalize

ylist  = LPYD[:,3].astype(int)  # Extract y-values from LPYD (with duplicates)
ylist2 = []                     # ylist 2 = ylilst without duplicates
for y in ylist:
    if y not in ylist2:
        ylist2.append(y)    
for k in range(len(ylist)):     # reduce values in ylist to 0 through Cases
    ylist[k] = ylist2.index(ylist[k])
    
Blist = LPYD[:,4].astype(int)   # list of boudary condition being considered
Dlist = LPYD[:,5].astype(int)   # list of coupling distances being considered
TempList = LPYD[:,6].astype(float)
Cases = len(plist)

N_s = PDG.N_Samples

############################################################################
##################### IMPORT AND SET PARAMETERS ############################
############################################################################

### Environment Geometry and Hamiltonian ###
LatLen = PDG.LatticeLength              # Number of qubit systems along each axis
N_Env = LatLen                 # Number of qubit systems in the environment

sigma = PDG.sigma           # Normalized smearing
wD = PDG.wD                    # Normalized Detector Gap   
mcc = PDG.mcc                 # Normalized Field Mass
lam = PDG.lam
                    
### Defining the Dynamics ###
PlotTimes = PDG.PlotTimes
N_times    = PDG.N_times				    # Number of time steps considered in each scenario

### Picking Environment States ###
TDist = PDG.ThermalStateDist        # Distribution thermal states are chosen from
TMean = PDG.TMean        # Parameter of this distribution                        
TDev  = PDG.TDev          # Parameter of this distribution
Gsignal= PDG.Gsignal                    # Squeezing of the "signal oscillator"

### Measurement Options ###
N_tom = PDG.N_tom                   # Number of tomography experiments to run in each direction at each time point


### PCA Options ###
RunPCAonData = PDG.RunPCAonData
PCA_Var_Keep = PDG.PCA_Var_Keep
N_PCA_plot   = PDG.N_PCA_plot      # Number of data points to plot in PCA 

### Classification Options ###
RunNNonData = PDG.RunNNonData
f_train = PDG.f_train
f_valid = PDG.f_valid
f_test = PDG.f_test
nH1    = PDG.nH1
nH2    = PDG.nH2
dropout_prob = PDG.dropout_prob
L2reg = PDG.L2reg
learning_rate     = PDG.learning_rate
N_epochs = PDG.N_epochs
minibatch_size = PDG.minibatch_size

############################################################################
##################### INITIALIZATION #######################################
############################################################################
print(' ')
print(' ')

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


print('Computing Hamiltonians')
Hlist_dynamic = [0]*Cases         # List of Hamiltonians for dynamics
Hlist_thermal = [0]*Cases         # List of Hamiltonians for thermality
for k in range(Cases):
    B = Blist[k]
    D = Dlist[k]
    Hlist_dynamic[k], Hlist_thermal[k] = u.ComputeHams(wD,mcc,lam,LatLen,sigma,B,D)
print('Done!')

print('Considering times:',PlotTimes)
output = np.zeros((5,len(PlotTimes)))
for t in range(len(PlotTimes)):
    if t == 0:
        continue
    print(' ')
    print(' ')
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
    print('Computing Unitaries and Evolved Projectors')
    ProjList = [0]*Cases
    for k in range(Cases):
        ProjList[k] = u.DefProjList(Hlist_dynamic[k],N_times,Tmin,Tmax)
    print('Done!')
    MedProj = np.median(ProjList,axis=0)
   
    ############################################################################
    ######################## COMPUTING EXACT STATISTICS ########################
    ############################################################################
                
    print('Precomputing Exact Trajectories')
    OutputFile = 'ExactSolutions_'+CaseString
    os.chdir(mycwd)
    try:
        os.mkdir(mycwd + '/' + OutputFile)
    except FileExistsError:
        pass
    
    os.chdir(mycwd + "/" + OutputFile)
    
    RS0 = u.InitializeProbeState('Ground')
    
    RSE0List = [0]*Cases
    for k in range(Cases):
        RE0 = u.ThermalState(Hlist_thermal[k],TempList[k]) # Compute the environment's thermal state
        if B == 3:
            RE0[0,0]= Gsignal
            RE0[0,N_Env]= 0
            RE0[N_Env,0]= 0
            RE0[N_Env,N_Env]= 1/Gsignal
    
        RSE0 = u.directsum(RE0,RS0)                       # Compute the initial probe-environment state
        RSE0List[k] = RSE0
    MedRSE0 = np.median(RSE0List,axis=0)

    d1 = MedProj.shape[0]
    d2 = MedProj.shape[1]
    MedAS = np.zeros((d1,d2))
    for n in range(d1):                                            
        for r in range(d2):
             MedAS[n,r] = np.trace(MedProj[n,r] @ MedRSE0).real
    MedAS = np.array(MedAS).flatten().real

    PrePickedAS = [0]*(Cases+1)
    PrePickedAS[Cases] = MedAS
    for k in range(Cases):
        
        dP = ProjList[k] - MedProj
        dS = RSE0List[k] - MedRSE0                
        d1 = ProjList[k].shape[0]
        d2 = ProjList[k].shape[1]
        aS = np.zeros((d1,d2))
        for n in range(d1):                                            
            for r in range(d2):
                 aS[n,r] += np.trace(dP[n,r] @ MedRSE0).real
                 aS[n,r] += np.trace(MedProj[n,r] @ dS).real                         
                 aS[n,r] += np.trace(dP[n,r] @ dS).real
        PrePickedAS[k] = np.array(aS).flatten().real

        print('Done for case',k+1,'of',Cases,'!')

    if TDev != 0 and t == 1:
        d1=RSE0List[0].shape[0]
        d2=RSE0List[0].shape[1]
        BigDSList = np.zeros((Cases*N_s,d1,d2))
        for c in range(Cases):
            print('Creating states for case',c+1,' of ',Cases,'!')
            for s in range(N_s):
                Temp = u.RTemp(TempList[c],TDev,'Uniform')
                RE0 = u.ThermalState(Hlist_thermal[c],Temp)
                RSE0 = u.directsum(RE0,RS0)
                BigDSList[c*N_s+s] = RSE0 - MedRSE0
                    
                if N_s < 20:
                    pass
                elif s % int(N_s/20) == 0:
                    print( 'States created:',s ,'Percent complete:', int(100*s/N_s) )
    
    print('Outputing Exact Trajectory Data to csv')
    #b2.save(OutputFile+'\ExactBlochTraj_All.pdf')
    pd.DataFrame(PrePickedAS).to_csv('ExactTrajData_'+CaseString+'.csv',header=None,index=None)
    print('Done!')

    ############################################################################
    ##################### HELLINGER DISTANCE & log(N(1/2)) ##############################
    ############################################################################
    if TDev != 0:
        print('Cannot Compute Hellinger Distance when TDev != 0')
    else:
        print('Computing Hellinger Distance')
        Hell = np.zeros((Cases,Cases))
        Nhalf = np.zeros((Cases,Cases))

        MedAS = PrePickedAS[Cases]            
        for c1 in range(Cases):
            for c2 in range(Cases):
                dmu = PrePickedAS[c1] - PrePickedAS[c2]
                sig1 = 2*(MedAS + PrePickedAS[c1])**2
                sig2 =  2*(MedAS + PrePickedAS[c2])**2                
                Hellinger, N_half = u.ComputeHellinger(dmu,sig1,sig2,N_tom-1)
                                
                Hell[c1,c2] = Hellinger
                Nhalf[c1,c2] = N_half
            
            DF_Hell = pd.DataFrame.from_items([(YLabels[k],Hell[k]) for k in range(len(YLabels))],orient='index', columns=YLabels)
            DF_Hell.to_csv('Hell_'+CaseString+'.csv')
            
        
    ############################################################################
    ################# SIMULATE EXPERIMENTAL DATA ###############################
    ############################################################################
    if GenerateData == True: 
        print('Creating data')
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
            print('Creating data for case',c+1,' of ',Cases,'!')
            aS = PrePickedAS[c]                                        # Look up the Bloch trajectory for this case
            extrainfo = np.array([ylist[c]])                           # Define extra information about this case
            LO = len(extrainfo)                                        # Define the "leave off" length

            if TDev != 0:
                dP = ProjList[c] - MedProj
                d1 = ProjList[c].shape[0]
                d2 = ProjList[c].shape[1]
                aS1 = np.zeros((d1,d2))
                for n in range(d1):
                    for r in range(d2):
                        aS1[n,r] += np.trace(dP[n,r] @ MedRSE0).real


            for s in range(N_s):                
                if TDev != 0:                
                    dP = ProjList[c] - MedProj
                    dS = BigDSList[c*N_s+s]
                    d1 = ProjList[c].shape[0]
                    d2 = ProjList[c].shape[1]
                    aS = np.zeros((d1,d2))
                    for n in range(d1):
                        for r in range(d2):
                            aS[n,r] += aS1[n,r]
                            aS[n,r] += np.trace(ProjList[c][n,r] @ dS).real                         
                    aS = np.array(aS).flatten().real
                    
                    
                aS_tom = u.Tomography(aS,N_tom,MedAS)                       # Add tomographic noise
                ExpData0[c,s] = np.concatenate((aS_tom,extrainfo), axis=None)
                ExpData[c*N_s+s] = np.concatenate((aS_tom,extrainfo), axis=None) # Save this trajectory        
                if N_s < 20:
                    pass
                elif s % int(N_s/20) == 0:
                    print( 'Data points created:',s ,'Percent complete:', int(100*s/N_s) )   
            print('Data points created:',N_s ,'Percent complete:', 100 )
            pd.DataFrame(ExpData0[c]).to_csv('Raw_TrajData_'+YLabels[c]+'.csv',header=None,index=None)
        
        print('Outputing Raw Trajectory Data to csv')
        pd.DataFrame(ExpData).to_csv('Raw_TrajData_All.csv',header=None,index=None)
        print('Done!')
        
        ExampleAS = np.zeros((Cases,3*N_times))
        for c in range(Cases):
            ExampleAS[c] = ExpData[c*N_s,:-1]
        
        os.chdir("..")
        
        print(mycwd)

    ############################################################################
    ################################ RUN PCA ON DATA ###########################
    ############################################################################

    if RunPCAonData == True:
        if GenerateData == False:
            OutputFile = CaseFile+'\ExpData_'+CaseString
            print('Loading in Data')
            ExpData = pd.read_csv(OutputFile+'\Raw_TrajData_All.csv',header=None).to_numpy()
            LO=1
            print('Loaded Data')

        os.chdir(mycwd)
        OutputFile = 'PCAdData_'+CaseString
        try:
            os.mkdir(OutputFile)
        except FileExistsError:
            pass        

        os.chdir(mycwd + "/"  + OutputFile)

        ### Randomly shuffle the indices ###
        print('Shuffling Data')
        permut = np.arange(ExpData.shape[0])
        np.random.shuffle(permut) 
        ExpData = ExpData[permut,:]
        X = ExpData[:,:-LO]
        y = ExpData[:,-1:]

    
        n_train = int(f_train*X.shape[0])  # Number of data points used for training
        n_valid = int(f_valid*X.shape[0])  # Number of data points used for validating     
    
        X_train = X[:n_train]            # X is the unlabeled data points 
        y_train = y[:n_train]            # Y is the labels 

        print('Running PCA on Training Data')    
        Xm, lam, M = u.PCA(X_train)                          # Run PCA on the data
        
        X = np.dot(X - np.tile(Xm, (X.shape[0], 1)), M.T).real
        X = X/np.sqrt(lam.T)
        PCAdData = np.append(X,y.reshape((len(y),1)),axis=1) # Add labels back to PCA'd data
    
        print('Outputing Eigensystem and PCA\'d Data to csv')
        pd.DataFrame(Xm).to_csv('PCA_Mean_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(lam).to_csv('PCA_EigVal_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(M).to_csv('PCA_EigVec_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(PCAdData).to_csv('PCAd_TrajData_'+CaseString+'.csv',header=None,index=None)
        print('Done!')


        d_0 = len(lam)
        if PCA_Var_Keep == 'All':
            d_c = d_0
        elif 0 < PCA_Var_Keep < 1:
            print('Compressing Data with PCA')    
            d_c = u.PCA_Compress(lam,PCA_Var_Keep)
            print('Input dimension compressed to',d_c,'of',d_0,'dimensions, keeping', 100*PCA_Var_Keep,'percent of the variance.')
        else:
            print('Removing extremely low variance dimensions')
            d_c = u.PCA_Compress(lam,1-10**(-10))
            print(d_0 - d_c,'of',d_0,'dimensions had extremely low variance and were removed.')
            
        d_c = max(d_c,Cases,3)
        M = M[:d_c,:]
        dlist = list(range(PCAdData.shape[1]))
        del dlist[d_c:-1]
        PCAdData = PCAdData[:,dlist]
        pd.DataFrame(M).to_csv('PCA_CutEigVec_TrajData_'+CaseString+'.csv',header=None,index=None)
        pd.DataFrame(PCAdData).to_csv('PCAd_CutTrajData_'+CaseString+'.csv',header=None,index=None)
    
        print('Plotting Data in Max Variance Plane')
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
        print('Done!')
    	
    
        os.chdir("..")
        print(mycwd)

    
    
    ############################################################################
    ##################### PROCESSING DATA FOR N.N. #############################
    ############################################################################
    
    if RunNNonData == True:
        print('Processing Data for Neural Network Training')
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
    
        xdata = x_train
        ydata = y_train
    
#        PCAdExamples = (np.array(ExampleAS)-np.tile(Xm, (Cases, 1))) @ M.T
        
        print('Done!')
        
        ############################################################################
        ##################### DEFINE THE NETWORK ARCHITECTURE ######################
        ############################################################################
        
        print('Defining Network Architecture')
        nI = xdata.shape[1]         # Input  dimension
        nO = len(YLabels)           # Output dimension
        
        ### Create placeholders for the input data and labels ###
        x = tf.placeholder(tf.float32, [None, nI]) # input data
        y = tf.placeholder(tf.int32,[None])       # labels
        
        ### Layer 1: ###
        n1i = nI
        n1o = nH1
        v1  = np.sqrt(2/(n1i+n1o))
        Init1 = tf.random_normal([n1i,n1o], mean=0.0, stddev=v1, dtype=tf.float32)
         
        W1 = tf.Variable(Init1)
        b1 = tf.Variable(tf.zeros([n1o]))
        z1 = tf.matmul(x, W1) + b1
        a1 = tf.nn.leaky_relu( z1 )        
        keep_prob = tf.placeholder("float")
        a1_drop = tf.nn.dropout(a1, keep_prob)
        
        ### Layer 2: ###
        n2i = n1o
        n2o = nO if nH2 == 'Skip' else nH2
        v2 = np.sqrt(2/(n2i+n2o))
        Init2 = tf.random_normal([n2i,n2o], mean=0.0, stddev=v2, dtype=tf.float32)

        W2 = tf.Variable(Init2)
        b2 = tf.Variable(tf.zeros([n2o]) )
        z2 = tf.matmul(a1, W2) + b2
        a2 = tf.nn.leaky_relu( z2 )
        
        ### Layer 3: ###
        if nH2 != 'Skip':
            n3i = n2o
            n3o = nO
            v3 = np.sqrt(2/(n3i+n3o))
            Init3 = tf.random_normal([n3i,n3o], mean=0.0, stddev=v3, dtype=tf.float32)

            W3 = tf.Variable(Init3)
            b3 = tf.Variable( tf.zeros([n3o]) )
            z3 = tf.matmul(a2, W3) + b3
            a3 = tf.nn.leaky_relu( z3 )
        
        aL = tf.nn.softmax(a2) if nH2 == 'Skip' else tf.nn.softmax(a3)    
        
        ### Cost function: ###
        y_onehot = tf.one_hot(y,depth = nO) # labels are converted to one-hot representation
        eps = 10**(-10) # to prevent the logs from diverging
        cross_entropy = tf.reduce_mean(-tf.reduce_sum( y_onehot * tf.log(aL+eps) +  (1.0-y_onehot )*tf.log(1.0-aL +eps) , reduction_indices=[1]))
        L2cost= L2reg*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2))
        if nH2 !='Skip': L2cost += tf.nn.l2_loss(W3)
        cost_func = cross_entropy+L2cost
        
        ### Use backpropagation to minimize the cost function using the gradient descent algorithm: ###
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_func)
        
        print('Done!')
        
        ##############################################################################
        ################################## TRAINING ##################################
        ##############################################################################
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        epoch_list    = [-1]
        cost_training = [np.nan]
        cost_valid    = [np.nan]
        acc_training  = [1/Cases]
        acc_valid     = [1/Cases]
        disp_list     = []    
        ### Train using mini-batches for several epochs: ###
        print("Beginning Training")
        
        permut = np.arange(n_train)
        num_iterations = 0
        for epoch in range(N_epochs+1):
            np.random.shuffle(permut) # Randomly shuffle the indices
            x_shuffled = x_train[permut,:]
            y_shuffled = y_train[permut]
        
            #Loop over all the mini-batches:
            for b in range(0, n_train, minibatch_size):
                x_batch = x_shuffled[b:b+minibatch_size,:]
                y_batch = y_shuffled[b:b+minibatch_size]
                sess.run(train_step, feed_dict={x: x_batch, y:y_batch, keep_prob:dropout_prob})
                num_iterations = num_iterations + 1
            
            ### Update the plot and print results every few epochs: ###
            if epoch % 1 == 0:
                cost_tr = sess.run(cost_func,feed_dict={x:x_train, y:y_train, keep_prob:1.0})
                cost_va = sess.run(cost_func,feed_dict={x:x_valid, y:y_valid, keep_prob:1.0})
                                
                NN_output_tr = sess.run(aL,feed_dict={x:x_train, y:y_train, keep_prob:1.0})
                predicted_class_tr = np.argmax(NN_output_tr, axis=1)
                acc_tr = np.mean(predicted_class_tr == y_train)
                
                NN_output_va = sess.run(aL,feed_dict={x:x_valid, y:y_valid, keep_prob:1.0})
                predicted_class_va = np.argmax(NN_output_va, axis=1)
                acc_va = np.mean(predicted_class_va == y_valid)
            
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
                disp_list.append(valid_matrix)
                
                ### Update the plot of the resulting classifier: ###
                if epoch % 50 == 0:
                    print( "Iteration %d:\n  Training cost %f \n  Validation cost %f\n  Training accuracy %f\n  Validating accuracy %f\n" % (epoch, cost_tr, cost_va, acc_tr, acc_va) )
                    Display = pd.DataFrame.from_items([(YLabels[k],valid_matrix[k]) for k in range(len(YLabels))],orient='index', columns=YLabels + ['Pres.'])
                    print(Display)
                    print(' ')
                    print(' ')
        
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
            
        fin_disp = np.mean(np.array(disp_list[int(0.8*len(disp_list)):]),axis=0)
        fin_acc = np.zeros((1,Cases+1)) 
        fin_acc[0,0] = np.mean(np.array(acc_valid[int(0.8*len(acc_valid)):]))
        fin_disp = np.append(fin_disp,fin_acc,axis=0)
        rows = list(YLabels)+['Acc.'] 
        fin_Disp = pd.DataFrame.from_items([(rows[k],fin_disp[k]) for k in range(len(rows))],orient='index', columns=YLabels+['Prec.'])
        fin_Disp.to_csv('Display_'+CaseString+'.csv')
       	
    if TDev == 0:

        output[0,t] = PlotTimes[t]
        #output[1,t] = np.mean(np.array(acc_valid[int(0.8*len(acc_valid)):]))
        #output[2,t] = np.std(np.array(acc_valid[int(0.8*len(acc_valid)):]))
        #output[2,t] = TV[0,1]
        output[3,t] = Hell[0,1]
        output[4,t] = Nhalf[0,1]
        #output[5,t] = Hell[1,2]
        #output[6,t] = Nhalf[1,2]
        
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
        #output[4,t] = np.mean(np.array(acc_valid[int(0.8*len(acc_valid)):]))
        
        os.chdir("..")
        os.chdir("..")
        
        np.save('BinaryStats', output)	
        pd.DataFrame(output).to_csv('BinaryStats.csv',header=None,index=None)

   