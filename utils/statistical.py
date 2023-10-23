#! /usr/bin/python
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import pandas as pd


###############################################################################
############################# DEFINITIONS #####################################
###############################################################################
#Order of appearance in datagenerator

def make_folder(base, folder_name):
    os.chdir(base)
    mycwd = os.getcwd()
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass

    os.chdir(mycwd + "/" + folder_name)
    mycwd = os.getcwd()

    return mycwd

def get_valid_train_test_from_data(PCAdData, nsamples, f_train, f_valid, LO, regression):
    n_train = int(f_train*nsamples)  # Number of data points used for training
    n_valid = int(f_valid*nsamples)  # Number of data points used for validating     
    x_train = PCAdData[:n_train,:-LO]
    x_valid = PCAdData[n_train:n_train+n_valid,:-LO]
    x_test  = PCAdData[n_train+n_valid:,:-LO]
    y_train = PCAdData[:n_train,-1:].flatten()
    y_valid = PCAdData[n_train:n_train+n_valid,-1:].flatten()
    y_test  = PCAdData[n_train+n_valid:,-1:].flatten()

    if regression == True:
        y_train = np.reshape(y_train,(y_train.shape[0],1))
        y_valid = np.reshape(y_valid,(y_valid.shape[0],1))
        y_test = np.reshape(y_test,(y_test.shape[0],1))

    return n_train, n_valid, x_train, x_valid, x_test, y_train, y_valid, y_test

def define_NN(nI, nO, nH1, regression, L2reg, lr):
    X = tf.placeholder("float", [None, nI])
    if regression == True:
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

    Y_hat=neural_net(X, QReg=regression)            

    L2cost = L2reg*(tf.nn.l2_loss(weights['h1'])+tf.nn.l2_loss(weights['out']))

    if regression == False:
        Y_onehot = tf.one_hot(Y,depth = nO)
        eps = 10**(-10) # to prevent the logs from diverging
        cross_entropy = tf.reduce_mean(-tf.reduce_sum( Y_onehot * tf.log(Y_hat+eps), reduction_indices=[1]))
        loss_op = cross_entropy + L2cost

    if regression == True:
        mse_cost = tf.losses.mean_squared_error(Y,Y_hat)
        loss_op = mse_cost + L2cost

    optimizer = tf.train.GradientDescentOptimizer(lr)#define optimizer # play around with learning rate
    train_op = optimizer.minimize(loss_op)#minimize losss

    return Y_hat, Y, X, train_op, loss_op

def train_validate_NN(nn_data: tuple, train_valid_data: tuple, reg_data: tuple, training_data: tuple, verbose: bool, YLabels):
    # nn_data = (Y_hat, Y, X, train_op, loss_op)
    # train_valid_data : (n_train, n_valid, x_train, x_valid, x_test, y_train, y_valid, y_test)
    # reg_data : (regression, minr, maxr, rdev)
    # training_data : (n0, minibatch_size, n_cases, N_epochs)

    Y_hat = nn_data[0]
    Y = nn_data[1]
    X = nn_data[2]
    train_op = nn_data[3]
    loss_op = nn_data[4]

    n_train = train_valid_data[0]
    n_valid = train_valid_data[1]
    x_train = train_valid_data[2]
    x_valid = train_valid_data[3]
    x_test  = train_valid_data[4]
    y_train = train_valid_data[5]
    y_valid = train_valid_data[6]
    y_test  = train_valid_data[7]

    reg  = reg_data[0]
    minr = reg_data[1]
    maxr = reg_data[2]
    rdev = reg_data[3]
    
    nO = training_data[0]
    minibatch_size = training_data[1]
    n_cases = training_data[2]
    N_epochs  = training_data[3]

    epoch_list    = [-1]
    cost_training = [np.nan]
    cost_valid    = [np.nan]
    acc_training  = [1/n_cases]
    acc_valid     = [1/n_cases]
    if reg == True:
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
                if reg == True:
                    temphat_tr = (NN_output_tr-0.25)*2*(maxr-minr)+minr
                    temphat_va = (NN_output_va-0.25)*2*(maxr-minr)+minr
                    temp_tr = (y_train-0.25)*2*(maxr-minr)+minr
                    temp_va = (y_valid-0.25)*2*(maxr-minr)+minr
                    acc_tr = np.mean(abs(temphat_tr - temp_tr) < rdev)
                    acc_va = np.mean(abs(temphat_va - temp_va) < rdev)
                else:
                    predicted_class_tr = np.argmax(NN_output_tr, axis=1)
                    acc_tr = np.mean(predicted_class_tr == y_train)
                    predicted_class_va = np.argmax(NN_output_va, axis=1)
                    acc_va = np.mean(predicted_class_va == y_valid)

                if reg == False:
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
                            g[s] == 1/n_cases
                        else:
                            g[s] = valid_matrix[s,s]/gs
                    valid_matrix = np.append(valid_matrix,g,axis=1)
                
                epoch_list.append(epoch)
                cost_training.append(cost_tr)
                cost_valid.append(cost_va)
                acc_training.append(acc_tr)
                acc_valid.append(acc_va)

                if reg == False: 
                    disp_list.append(valid_matrix)
            if type(verbose) == int:
                if verbose > 0:
                    if epoch % verbose == 0:
                        print( "Iteration %d:\n  Training cost %f \n  Validation cost %f\n  Training accuracy %f\n  Validating accuracy %f\n" % (epoch, cost_tr, cost_va, acc_tr, acc_va) )
                        if reg == False:
                            Display = pd.DataFrame.from_dict(dict([(YLabels[k],valid_matrix[k]) for k in range(len(YLabels))]),orient='index', columns=YLabels + ['Pres.'])
                            print(Display)
                        print(' ')
                        print(' ')

    return acc_valid

def PCA(X=np.array([])):
    (n, d) = X.shape                                  # Determine data shape

    Xm = np.mean(X, 0).real                           # Compute the mean of each colum
    X0  = X - np.tile(Xm, (n, 1))                     # Subtract off mean from each row
    Cov = np.dot(X0.T, X0)/(n-1)

    (lam, M) = np.linalg.eig(Cov)  # Compute eigensystem
    
    lam = lam.real               
    M = M.real    
    idx = lam.argsort()[::-1]       
    lam = lam[idx]
    M = M[:,idx]
    M=M.T
    
    for k in range(len(lam)):
        if k == 0:
            continue
        if lam[k] < 0:
            lam[k] = lam[k-1]/10
            
    return Xm, lam, M

def PCA_Compress(lam,VarKeep):
    sum_lam = sum(lam)
    cumsum = 0
    for dim_c in range(len(lam)):
        cumsum += lam[dim_c]
        if cumsum/sum_lam > VarKeep:
            break
    dim_c += 1
    return dim_c

def run_PCA_on_data(data, LO, f_train, n_cases, PCA_var_keep):
    ### Randomly shuffle the indices ###
    permut = np.arange(data.shape[0])
    np.random.shuffle(permut) 
    data = data[permut,:]
    X = data[:,:-LO]
    y = data[:,-1:]

    n_train = int(f_train*X.shape[0])  # Number of data points used for training
    X_train = X[:n_train]            # X is the unlabeled data points 

    Xm, lam, M = PCA(X_train)                          # Run PCA on the data
    X = np.dot(X - np.tile(Xm, (X.shape[0], 1)), M.T).real
    X = X/np.sqrt(lam.T)
    PCAdData = np.append(X,y.reshape((len(y),1)),axis=1) # Add labels back to PCA'd data

    d_0 = len(lam)
    if PCA_var_keep == 'All':
        d_c = d_0
    elif 0 < PCA_var_keep < 1:
        d_c = PCA_Compress(lam,PCA_var_keep)
    else:
        d_c = PCA_Compress(lam,1-10**(-10))
        
    d_c = max(d_c,n_cases,3)
    dlist = list(range(PCAdData.shape[1]))
    del dlist[d_c:-1]
    PCAdData = PCAdData[:,dlist]

    return PCAdData

def ComputeHellinger(dmu,sig1,sig2,N_tom):
    Log1 = 0
    Log2 = 0    
    for k in range(dmu.shape[0]):
        Log1 += (1/8)*dmu[k]**2/((sig1[k]+sig2[k])/2)
        Log2 += (1/4)*np.log(sig1[k])
        Log2 += (1/4)*np.log(sig2[k])
        Log2 += -(1/2)*np.log(((sig1[k]+sig2[k])/2))
                
    Hellinger = np.sqrt(1-np.exp(Log2 - Log1*N_tom))
    N_half = (np.log(Log2 -np.log(1/2)) - np.log(Log1))/np.log(10)

    return Hellinger, N_half

