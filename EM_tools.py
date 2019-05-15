#!/usr/bin/env python3
import numpy as np

def gamma_expectation(train_index , test_index , probs , gamma):
    denoms = np.sum(probs[np.ix_(test_index , train_index)] , axis = 1)
    gamma[np.ix_(test_index , train_index)] = probs[np.ix_(test_index , train_index)] / denoms[:,None]
    return(gamma)


def gamma_expectation_log(train_index , test_index , probs , gamma):
    logC = np.max(probs[np.ix_(test_index , train_index)],1).reshape(len(test_index) , 1)
    denoms = np.sum(np.exp(probs[np.ix_(test_index , train_index)] - logC) , axis = 1) 
    num = np.exp(probs[np.ix_(test_index , train_index)] - logC) # Mulig fejl
    gamma[np.ix_(test_index , train_index)] = num  / denoms[:, None]
    return(gamma)

    
def sigma_maximize_no_normalization(train_index , test_index , X , gamma , dim ):
    sigma = np.zeros((dim , dim))
    for i in test_index:
        N_train = len(train_index)
        x_test = X[i]
        #x_train = X[n]
        diff = x_test - X[train_index]      
        sigma += np.dot(np.transpose(gamma[i, train_index].reshape(N_train, 1) * diff), diff)
    return(sigma)