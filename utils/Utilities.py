import numpy as np
import pandas as pd
import utils.dynamics as d
import utils.statistical as stat

def get_probe_trajectories(
    Flist_dynamic,
    Flist_thermal,
    LatLenList,
    Blist,
    TempList,
    TDev,
    tmin,
    tmax,
    t_index,
    measurements_per_window,
    cases,
    n_samples,
    Regression, 
    Gsignal=1):

    projector_list = [0]*cases
    for k in range(cases):
        projector_list[k] = d.getProjectorList(Flist_dynamic[k],measurements_per_window,tmin,tmax)
    median_projector = np.median(projector_list,axis=0) # Compute the median projector
    
    print('Calculating states')

    probe_state_0 = d.InitializeProbeState('Ground')
    state_list_0 = [0]*cases                                   # List of probe-environment initial states
    for k in range(cases):
        # below is setup dependent
        phi_state_0 = d.ThermalState(Flist_thermal[k],TempList[k]) # Compute the environment's thermal state

        
        if Blist[k] == 3:                                         # In the signaling case
            phi_state_0[0,0]= Gsignal                              # Squeeze the last oscillator
            phi_state_0[0,LatLenList[k]]= 0
            phi_state_0[LatLenList[k],0]= 0
            phi_state_0[LatLenList[k],LatLenList[k]]= 1/Gsignal
        state_list_0[k] = d.directsum(phi_state_0,probe_state_0)                 # Compute the initial probe-environment state
    median_state_0 = np.median(state_list_0,axis=0)                   # Compute the median probe-environment stat

    ### Computing "Median" Probe Trajectory ###
    ### ^Purely for ease of calculation and computation ###
    d1 = median_projector.shape[0]
    d2 = median_projector.shape[1]
    median_trajectory = np.zeros((d1,d2))
    for n in range(d1):                                            
        for r in range(d2):
                median_trajectory[n,r] = np.trace(median_projector[n,r] @ median_state_0).real
    median_trajectory = np.array(median_trajectory).flatten().real


    print('Calculating probe trajectories')
    ### Calculating probe trajectory in for each case
    prepicked_trajectories = [0]*(cases+1)
    prepicked_trajectories[cases] = median_trajectory
    for k in range(cases):        
        dP = projector_list[k] - median_projector      # Difference from median projector
        dS = state_list_0[k] - median_state_0      # Difference from median state
        #assert(np.count_nonzero(dS) > 0)
        trajectory = np.zeros((d1,d2))
        for n in range(d1):             # Compute difference from "median" trajectory
            for r in range(d2):
                    trajectory[n,r] += np.trace(dP[n,r] @ median_state_0).real
                    trajectory[n,r] += np.trace(median_projector[n,r] @ dS).real
                    trajectory[n,r] += np.trace(dP[n,r] @ dS).real

        #assert(np.count_nonzero(trajectory) > 0)
        prepicked_trajectories[k] = np.array(trajectory).flatten().real

    ### Picking Random initial states for thermal case ###
    if Regression == True: 
        reglist = np.zeros((cases*n_samples,))
    else:
        reglist = []

    if TDev != 0 and t_index == 1:
        d1=state_list_0[0].shape[0]
        d2=state_list_0[0].shape[1]
        BigDSList = np.zeros((cases*n_samples,d1,d2))
        for c in range(cases):
            for s in range(n_samples):
                Temp = d.RTemp(TempList[c],TDev)
                phi_state_0 = d.ThermalState(Flist_thermal[c],Temp)
                state_0 = d.directsum(phi_state_0,probe_state_0)
                BigDSList[c*n_samples+s] = state_0 - median_state_0
                if Regression == True: reglist[c*n_samples+s] = Temp
    else:
        BigDSList = []

    if Regression == True and t_index == 1:
        minr = min(reglist)
        maxr = max(reglist)
        reglist = reglist - minr
        reglist = reglist/(maxr-minr)
        reglist = 0.5*reglist
        reglist = reglist + 0.25

    return projector_list, median_projector, state_list_0, median_state_0, prepicked_trajectories, median_trajectory, BigDSList, reglist


def generate_measurement_data(
    projector_list, 
    median_projector, 
    prepicked_trajectories, 
    median_state_0, 
    BigDSList,
    cases, 
    n_samples, 
    measurements_per_window, 
    n_tom, 
    TDev,
    ylist,
    Regression, 
    reglist,
    save=True, 
    path=''
    ):
    # generate data
    exp_data = np.zeros((cases*n_samples,3*measurements_per_window+1))
    print('Generating data')

    MedAS = prepicked_trajectories[cases]
    for c in range(cases):
        #print('Creating data for case',c+1,' of ',cases)
        aS = prepicked_trajectories[c]                                        # Look up the exact trajectory for this case
        extrainfo = np.array([ylist[c]])                           # Define extra information about this case

        LO = len(extrainfo)                                        # Define the "leave off" length
        # In thermal case compute environment state independent part of the probe trajectory
        if TDev != 0:                                              
            dP = projector_list[c] - median_projector
            d1 = projector_list[c].shape[0]
            d2 = projector_list[c].shape[1]
            aS1 = np.zeros((d1,d2))
            for n in range(d1):
                for r in range(d2):
                    aS1[n,r] += np.trace(dP[n,r] @ median_state_0).real

        for s in range(n_samples):                
            # In thermal case compute environment state dependent part of the probe trajectory
            if TDev != 0:                
                dS = BigDSList[c*n_samples+s]
                d1 = projector_list[c].shape[0]
                d2 = projector_list[c].shape[1]
                trajectory = np.zeros((d1,d2))
                for n in range(d1):
                    for r in range(d2):
                        trajectory[n,r] += aS1[n,r]
                        trajectory[n,r] += np.trace(projector_list[c][n,r] @ dS).real                         
                trajectory = np.array(trajectory).flatten().real
                if Regression == True: 
                    extrainfo = np.array([reglist[c*n_samples+s]])
            
            # Add tomographic noise
            trajectory_tom = d.Tomography(aS,n_tom,MedAS) 
            exp_data[c*n_samples+s] = np.concatenate((trajectory_tom,extrainfo), axis=None) # Save this trajectory        
    
    print('Done creating data')
    df = pd.DataFrame(exp_data)
    if save:
        df.to_csv(path+'exp_data_all.csv',header=None,index=None)

    return df


def run_PCA_on_data(exp_data, f_train, PCA_var_keep, cases, save=False, path=''):
    ### Randomly shuffle the indices ###
    LO=1
    np.random.shuffle(exp_data)
    X = exp_data[:,:-LO]
    y = exp_data[:,-1:]

    n_train = int(f_train*X.shape[0])  # Number of data points used for training
    X_train = X[:n_train]            # X is the unlabeled data points 

    print('Running PCA on Training Data')    
    Xm, lam, M = stat.PCA(X_train)                          # Run PCA on the data
    
    X = np.dot(X - np.tile(Xm, (X.shape[0], 1)), M.T).real
    X = X/np.sqrt(lam.T)
    PCAdData = np.append(X,y.reshape((len(y),1)),axis=1) # Add labels back to PCA'd data

    d_0 = len(lam)
    if PCA_var_keep == 'All':
        d_c = d_0
    elif 0 < PCA_var_keep < 1:
        d_c = stat.PCA_Compress(lam,PCA_var_keep)
    else:
        d_c = stat.PCA_Compress(lam,1-10**(-10))

    print('Saving PCA data') 
    d_c = max(d_c, cases, 3)
    M = M[:d_c,:]
    dlist = list(range(PCAdData.shape[1]))
    del dlist[d_c:-1]
    PCAdData = PCAdData[:,dlist]
    np.random.shuffle(PCAdData)
    if save:
        pd.DataFrame(PCAdData).to_csv(path+'pca_data_all.csv',header=None,index=None)

    return PCAdData
