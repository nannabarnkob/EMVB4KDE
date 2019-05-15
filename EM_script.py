# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:28:29 2019

@author: Mads
"""


kFold = 2
maxit = 100
tol = 10e-9

X = pd.read_csv('OldFaithful.csv')
X = X.values

#X_train, X_test = train_test_split(X, test_size=0.4, random_state=0)


N_all = len(X)


dim = np.shape(X)[1]
sigma = np.identity(dim)

kf = KFold(n_splits=kFold)


gamma = np.zeros((N_all , N_all))
probs = np.zeros((N_all , N_all))
denoms = np.zeros((N_all , 1))

iteration = 0
converged = False

likelihood_arr = np.array([])
while( iteration < maxit and (not converged)  ):
    
    iteration += 1
    sigma_old = sigma
    
    # Expectation
    print("Min")
    for n in range(N_all):
        probs[:,n] = multivariate_normal.pdf(X , X[n] , sigma_old)
    
    likelihood_arr = np.append(prop_arr , np.sum(probs))
    
    for k , (train_index, test_index) in enumerate(kf.split(X)): 
        denoms = np.sum(probs[np.ix_(test_index , train_index)] , axis = 1)
        gamma[np.ix_(test_index , train_index)] = probs[np.ix_(test_index , train_index)] / denoms[:,None]
    
    
    # Maximization
    print("Max")
    sigma = np.zeros((dim , dim))
    for k , (train_index, test_index) in enumerate(kf.split(X)):
        print(k)

        
        for i in test_index:
            N_test = len(test_index)
            x_test = X[i]
            #x_train = X[n]
            diff = x_test - X[train_index]      
            sigma += np.dot ( np.transpose(gamma[i,train_index].reshape(N_test,1) * diff) , diff)
    
    sigma = 1/N_all * sigma
    
    likelihood_diff = np.abs(likelihood_arr[iteration] - likelihood_arr[iteration-1])
    converged = likelihood_diff < tol
    
    print(iteration , likelihood_diff)