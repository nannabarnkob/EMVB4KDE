# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:22:29 2019

@author: Mads
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:24:53 2019

@author: Mads
"""

import numpy as np
from sklearn.model_selection import KFold , train_test_split , LeaveOneOut
from scipy.stats import multivariate_normal
from EM_tools import gamma_expectation_log , sigma_maximize_no_normalization




def EM_KernelDensity(X, cross_method, MaxIterations=100 , tol=10e-3 ,  Kfolds=10):
    #X_df = pd.read_csv(filename)
    #X = X_df.values
    
    
    N_all = len(X)
    if cross_method == "HoldOut":
        X_train, X_test, train_index, test_index = train_test_split(
        X, range(N_all), test_size=0.4, random_state=1)
        N_train, D_train = X_train.shape
        N_test, D_test = X_test.shape
        N_div = N_test
        
    elif cross_method == "Kfold":
        kf = KFold(n_splits=Kfolds)
        N_div = N_all

    elif cross_method == "LeaveOneOut":
        kf = LeaveOneOut()
        N_div = N_all
    
    
    dim = np.shape(X)[1]
    #sigma = 50 * np.identity(dim)

    sigma = np.cov(X, rowvar=False) # start guess is covariance matrix of data
    
    gamma = np.zeros((N_all, N_all))
    probs = np.zeros((N_all, N_all))
    
    iteration = 0
    converged = False
    
    likelihood_arr = np.array([0.0])
    
    normal_const_without_det = 2*np.math.pi**(dim/2)
    
    while( iteration < MaxIterations and (not converged)  ):
        #print(iteration)
        
        iteration += 1
        sigma_old = sigma

        # Expectation
        for n in range(N_all):
            probs[:, n] = multivariate_normal.pdf(X, X[n], sigma_old)
        
        # Expectation
        
        chol = np.linalg.cholesky(sigma)
        R = np.transpose(chol)

        A = np.dot(X , np.linalg.inv(R))
        
        squaredA = np.sum(np.multiply(A, A), 1).reshape((-1,1))
            
        if cross_method == "HoldOut":
            gamma = gamma_expectation_log(train_index , test_index , probs , gamma)
            propbs_exp = np.exp(probs)
            likelihood_arr = np.append(likelihood_arr , np.log(np.sum(propbs_exp)))

        elif cross_method == "Kfold" or cross_method == "LeaveOneOut" :
            for k , (train_index, test_index) in enumerate(kf.split(A)): 
                
                A_train = A[train_index,:]
                A_test = A[test_index,:]
                
                part1 = squaredA[test_index]
                part2 = np.dot(A_test , np.transpose(A_train))
                part3 = np.transpose(squaredA[train_index])
                probs[np.ix_(test_index , train_index)] = part1 - 2*part2 + part3
                
            
            
            const_log = np.log(normal_const_without_det * np.linalg.det(R))
            probs = -1/2 * probs - const_log
            propbs_exp = np.exp(probs)
            
            likelihood_arr = np.append(likelihood_arr , np.log(np.sum(propbs_exp)))
            
            for k, (train_index, test_index) in enumerate(kf.split(X)):
                gamma = gamma_expectation_log(train_index , test_index , probs , gamma)
             

        # Maximization
        if cross_method == "HoldOut":
            sigma = sigma_maximize_no_normalization(train_index , test_index , X , gamma , dim )

        elif cross_method == "Kfold" or cross_method == "LeaveOneOut":
            sigma = np.zeros((dim , dim))
            for k , (train_index, test_index) in enumerate(kf.split(X)):
                sigma += sigma_maximize_no_normalization(train_index , test_index , X , gamma , dim )

        # Normalize the maximized sigma
        sigma = 1/N_div * sigma

        # Convergence check
        if iteration >= 1:
            likelihood_diff = np.abs(likelihood_arr[iteration] - likelihood_arr[iteration-1])
            converged =  likelihood_diff < tol
    
        # print(iteration , likelihood_diff)
    return sigma, iteration, likelihood_arr


