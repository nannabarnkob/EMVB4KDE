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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold , train_test_split , LeavePOut
from scipy.stats import multivariate_normal

from scipy.special import binom

from EM_tools import gamma_expectation , sigma_maximize_no_normalization

import pandas as pd



def EM_KernelDensity(filename , cross_method , maxit=100, tol = 10e-3 ,  Kfolds = 2 , pOut = 1):
    
    
        

    X_df = pd.read_csv(filename)
    X = X_df.values
    
    
    N_all = len(X)
    
    if cross_method == "HoldOut":
        X_train, X_test = train_test_split(X_df, test_size=0.33, random_state=42)
        train_index = X_train.index.values
        test_index = X_test.index.values
        N_test = len(X_test)
        N_train = len(X_train)
        N_div = N_test
        
    elif cross_method == "Kfold":
        kf = KFold(n_splits=Kfolds)
        N_div = N_all
    elif cross_method == "Leave1Out":
        kf = LeavePOut(pOut)
        N_div = N_all
    
    
    dim = np.shape(X)[1]
    sigma = 50 * np.identity(dim)
    
    
    gamma = np.zeros((N_all , N_all))
    probs = np.zeros((N_all , N_all))
    denoms = np.zeros((N_all , 1))
    
    iteration = 0
    converged = False
    
    likelihood_arr = np.array([0.0])
    while( iteration < maxit and (not converged)  ):
        
        iteration += 1
        sigma_old  = sigma
        
        R = np.linalg.cholesky(sigma_old)
        R_inv = np.linalg.inv(R)
                
        # Expectation
        for n in range(N_all):
            probs[:,n] = multivariate_normal.pdf(X , X[n] , sigma_old)
            
        likelihood_arr = np.append(likelihood_arr , np.log(np.sum(probs)))
            
        if cross_method == "HoldOut":
            gamma = gamma_expectation(train_index , test_index , probs , gamma)
        # Probles with leave2out - Hvad gør man når hvert element er med i flere folds?? Genemsnit?
        
        elif cross_method == "Kfold" or cross_method == "LeavePOut" :
            for k , (train_index, test_index) in enumerate(kf.split(X)): 
                gamma = gamma_expectation(train_index , test_index , probs , gamma)
             

        # Maximization
        if cross_method == "HoldOut":
            sigma = sigma_maximize_no_normalization(train_index , test_index , X , gamma , dim )
        
        
        elif cross_method == "Kfold" or cross_method == "LeavePOut":
            sigma = np.zeros((dim , dim))
            for k , (train_index, test_index) in enumerate(kf.split(X)):
                sigma += sigma_maximize_no_normalization(train_index , test_index , X , gamma , dim )
            
       # elif cross_method == "LeavePOut":
       #     t = 2+2
            # Probles with leave2out - Hvad gør man når hvert element er med i flere folds?? Genemsnit?
            
            
            
        # Normaliza the maximized sigma
        sigma = 1/N_div * sigma
        
        
        # Convergence check
        likelihood_diff = np.abs(likelihood_arr[iteration] - likelihood_arr[iteration-1])
        converged =  likelihood_diff < tol
    
        print(iteration , likelihood_diff)
        
    return(sigma,iteration  , likelihood_arr)

