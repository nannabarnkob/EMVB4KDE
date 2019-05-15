import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import whosmat
import pandas as pd
from scipy.stats import multivariate_normal, wishart, zscore
from scipy.special import digamma
import math
import time as time
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.special import multigammaln
import pdb
from sklearn.utils import shuffle
import h5py


def logdet(Lambda):
  (s, ulogdet) = np.linalg.slogdet(Lambda)
  if abs(ulogdet == float("inf")):
      ulogdet = 0
  return s*ulogdet

def R_entropy(R):
  index = np.where(R != 0)
  R_Return = R[index]
  return(R_Return)
  
def dfun(x,y):
    return( (x-y).sum() )
  
def VB_inference(data,method,MaxIterations=300,showplot = True,k = 10,class_index = -1 , tol = 10e-3):
      #Check to see if file exists
#    try:
 #       fh = open(filename, 'r')
        # Store configuration file values
 #   except FileNotFoundError:
 #       print("The file doesn´t exist. Check to see if it lies in the directory")
 #      return()



    ## Load data
  #  if filename.endswith('.csv'):
  #      data = pd.read_csv(filename)
  #      data = data.as_matrix()
  #  elif filename.endswith('.mat'):
  #      content = whosmat(filename)
  #      
  #      full_data = loadmat(filename)
  #      data = full_data[content[0][0]]
  #  elif filename.endswith('.txt'):
  #      data = np.loadtxt(filename)
        
    #data = data[0:100,:]
    #print(data.shape)
   # ind = shuffle(range(272))
   # data = data[ind,:]
    
    
    
    #Get number of differnet classes
 
    if class_index == -1:
      N_class = 1
    else:
      classes = np.unique(data[:,class_index+1])
      N_class = len(classes)
      
    #print(classes)
    for q in range(N_class):
      Return_object = []
      
      if class_index == -1:
        X = data
      else:
        X = data[data[:,class_index+1] == classes[q],:]
      #Get dimensions
      N, D = X.shape
      
      #Transform data into matrix-format
      
      ## Priors
      v0 = 0.1
      
      try:
        W0 = np.linalg.inv((np.cov(X,rowvar = False))*v0) # Wishart covariance prior
        
      except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
          RN = np.random.normal(0,0.1,size = np.array([D]))
          W0 = np.linalg.inv(((np.eye(D)*RN+np.cov(X,rowvar = False))*v0)) # Wishart covariance prior
        else:
          raise
      W_k = W0
  
      #Number of iterations
      T = MaxIterations
      DM = np.empty((N,N,D))
      for i in range(D):
        #print(i)
        DM[:,:,i] = 1/2 * squareform(pdist(np.column_stack((X[:,i],X[:,i])) , metric = dfun))
      
      

      #Create cache
      hdf5_store = h5py.File("./cache.hdf5", "a")
      Outer_DM = hdf5_store.create_dataset("results", (D,D,(N)**2), compression="gzip")

      
      #Create a 3d-array with outerproduct of the pairwise differences
     # Outer_DM = np.empty((D,D,(N)**2))
      print("Create a 3d-array with outerproduct of the pairwise differences")
      for i in range(N):
        Holder = np.empty((D,D,N))
        for j in range(N):
          #print(j)
          diff = DM[i,j,:]
          Holder[:,:,j] = np.outer(diff,diff)
        print(np.max(Holder))
        Outer_DM[:,:,np.arange((i*(N)),(i+1)*N)] = Holder
          
      #Check if the method is holdout
      if method == "HoldOut":
          X_train, X_test, train_index, test_index = train_test_split(
          X, range(N), test_size=0.4, random_state=1)
          N_train, D_train = X_train.shape
          N_test, D_test = X_test.shape
      
      if method == "Kfold":
          kf = KFold(n_splits=k,shuffle=False)
      
      
      ELBO = np.array([])
      
      E_lnpX_arr = np.array([])
      E_lnqZ_arr = np.array([])
      E_lnpZ_arr = np.array([])
      E_lnqLambda_arr = np.array([])
      E_lnpLambda_arr = np.array([])
      E_Lambda_arr = np.array([])
      
      #print(E_lnpX)
          #print(E_lnqZ)
          #print(E_lnpZ)
          #print(E_lnqLambda)
          #print(E_lnpLambda)
      
      
      
      converged = False
      n = 0
      
      while not converged and (n < MaxIterations):
          # pdb.set_trace()
          #Print the current iteration
          print(n)
          
          ## Variational E-step
          #Refine W
          W = W_k
      
          #Allocate space to save responsibilities 
          R = np.zeros(np.array([N,N]))
          
          E_lnqZ = 0
          E_lnpZ = 0
          E_lnpX = 0
          weight_diff = 0
          #Check if the method is leave one out
          if method == "LeaveOneOut":
              
              loo = LeaveOneOut()
              for train_index, test_index in loo.split(X):
                  X_train, X_test = X[train_index], X[test_index]
                  
                  #Dimensions of train and test data
                  N_train, D_train = X_train.shape
                  N_test, D_test = X_test.shape
                  
                  v = v0 + N
                  
                  
                  E_Lambda = sum(digamma((v+1-(np.arange(D)+1))/2))+ D*math.log(2)+logdet(W)
                  E_pi = np.log(1/N_train)
                  
          
                  for i in range(N_test):
                      ln_rho = E_pi + 1/2*E_Lambda - D/2*math.log(2*math.pi) - \
                      1/2*v*np.sum(np.dot((X_test[i,:] - X_train),W)*(X_test[i,:] - X_train),axis = 1)
                      
                      ln_C = -max(ln_rho)
                      
                      R[test_index[i],train_index] = np.exp(ln_rho+ln_C)/sum(np.exp(ln_rho+ln_C))
                      R_fixed = R_entropy(R[test_index[i],train_index])
                      #E_lnqZ += sum(np.log(R_fixed)*R_fixed)
                      #E_lnpZ += sum(np.log(1/N_train)*R_fixed)
                      r = R[test_index[i],train_index]
  
                      #E_lnpX += 1/2*sum(r*(E_Lambda-v* \
                      # np.sum(np.dot((X_test[i,:] - X_train),W)*(X_test[i,:] - X_train),axis = 1) - D*math.log(2*math.pi)))
                      
          #Check if the method is Kfold
          elif method == "Kfold":
              
              for train_index, test_index in kf.split(X):
                  X_train, X_test = X[train_index], X[test_index]
                  
                  #Dimensions of train and test data
                  N_train, D_train = X_train.shape
                  N_test, D_test = X_test.shape
                  
                  v = v0 + N
                  
                  
                  E_Lambda = sum(digamma((v+1-(np.arange(D)+1))/2))+ D*math.log(2)+logdet(W)
                  E_pi = np.log(1/N_train)
                  
          
                  for i in range(N_test):
                      ln_rho = E_pi + 1/2*E_Lambda - D/2*math.log(2*math.pi) - \
                      1/2*v*np.sum(np.dot((X_test[i,:] - X_train),W)*(X_test[i,:] - X_train),axis = 1)
                      
                      ln_C = -max(ln_rho)
                      R[test_index[i],train_index] = np.exp(ln_rho+ln_C)/sum(np.exp(ln_rho+ln_C))
                      R_fixed = R_entropy(R[test_index[i],train_index])
                      #E_lnqZ += sum(np.log(R_fixed)*R_fixed)
                      #E_lnpZ += sum(np.log(1/N_train)*R_fixed)
                      #r = R[test_index[i],train_index]
  
                      #E_lnpX += 1/2*sum(r*(E_Lambda-v* \
                      # np.sum(np.dot((X_test[i,:] - X_train),W)*(X_test[i,:] - X_train),axis = 1) - D*math.log(2*math.pi)))
          #Check if the method is HoldOut
          elif method == "HoldOut":
              v = v0 + N_test
              
              E_Lambda = sum(digamma((v+1-(np.arange(D)+1))/2))+ D*math.log(2)+logdet(W)
              E_pi = np.log(1/N_train)
              
              for i in range(N_test):
                  ln_rho = E_pi + 1/2*E_Lambda - D/2*math.log(2*math.pi) - \
                  1/2*v*np.sum(np.dot((X_test[i,:] - X_train),W)*(X_test[i,:] - X_train),axis = 1)
                  
                  ln_C = -max(ln_rho)
                  R[test_index[i],train_index] = np.exp(ln_rho+ln_C)/sum(np.exp(ln_rho+ln_C))
                  R_fixed = R_entropy(R[test_index[i],train_index])
                  E_lnqZ += sum(np.log(R_fixed)*R_fixed)
                  E_lnpZ += sum(np.log(1/N_train)*R_fixed)
                  r = R[test_index[i],train_index]
                  
                  E_lnpX += 1/2*sum(r*(E_Lambda-v* \
                       np.sum(np.dot((X_test[i,:] - X_train),W)*(X_test[i,:] - X_train),axis = 1) - D*math.log(2*math.pi)))
    
          else:
              print("Could recognize the specified method. Pls check for spelling errors")
              return()
     
         
         # A = np.zeros((2,2)) 
          #Calculate the inverse of W
          for train_index, test_index in kf.split(X):
                  X_train, X_test = X[train_index], X[test_index]
                  
                  #Dimensions of train and test data
         #         N_train, D_train = X_train.shape
         #         N_test, D_test = X_test.shape
                  
         #         for k in range(N_train):
         #             for i in range(N_test):
         #                 A += R[test_index[i] , train_index[k]] * np.outer(X[test_index[i]] - X[train_index[k]] , X[test_index[i]] - X[train_index[k]])
          
         # W_inv_new_A = np.linalg.inv(W0) + A
          W_inv_new = np.linalg.inv(W0) + np.sum(np.reshape(R,np.array([1,N**2]))*Outer_DM[:,:,:],axis = -1)
          #print(W_inv_new_A)
          
          #Update W_k
          W_k = np.linalg.inv(W_inv_new)
          
          ## ASSUME KFOLD!!!
          #pdb.set_trace()
          for train_index, test_index in kf.split(X):
                  X_train, X_test = X[train_index], X[test_index]
                  
                  #Dimensions of train and test data
                  N_train, D_train = X_train.shape
                  N_test, D_test = X_test.shape
                  
                  v = v0 + N
                  
                  
                  E_Lambda = sum(digamma((v+1-(np.arange(D)+1))/2))+ D*math.log(2)+logdet(W_k)
                  E_pi = np.log(1/N_train)
                  
          
                  for i in range(N_test):
                      R_fixed = R_entropy(R[test_index[i],train_index])
                      E_lnqZ += sum(np.log(R_fixed)*R_fixed)
                      E_lnpZ += sum(np.log(1/N_train)*R_fixed)
                      r = R[test_index[i],train_index]
                      
                      weight_diff += sum(np.sum(np.dot((X_test[i,:] - X_train),W_k)*(X_test[i,:] - X_train),axis = 1))
  
                      E_lnpX += 1/2*sum(r*(E_Lambda-v * \
                       np.sum(np.dot((X_test[i,:] - X_train),W_k)*(X_test[i,:] - X_train),axis = 1) - D*math.log(2*math.pi)))
          
          print(weight_diff)
          E_lnqLambda = (v-D-1)/2*(sum(digamma((v+1-(np.arange(D)+1))/2))+ D*math.log(2)+logdet(W_k)) -1/2*v*D - \
          (v*D/2*math.log(2)+v/2*logdet(W_k)+multigammaln(v/2,D))
          
          E_lnpLambda = (v0-D-1)/2*(sum(digamma((v+1-(np.arange(D)+1))/2))+ D*math.log(2)+logdet(W_k)) - \
          1/2*v*np.matrix.trace(np.dot(np.linalg.inv(W0),W_k)) - \
          (v0*D/2*math.log(2)+v0/2*logdet(W0)+0)
          
          
          
          #pdb.set_trace()
          
          
          #print(E_lnpX)
          #print(E_lnqZ)
          #print(E_lnpZ)
          #print(E_lnqLambda)
          #print(E_lnpLambda)
          
          #ELBO = np.append(ELBO,np.array([E_lnpX - E_lnqZ + E_lnpZ  - E_lnqLambda]))
          ELBO = np.append(ELBO,np.array([E_lnpX - E_lnqZ + E_lnpZ - E_lnqLambda + E_lnpLambda]))
          print(ELBO[n])
          print(W_k)
          
          E_lnpX_arr = np.append(E_lnpX_arr , E_lnpX)
          E_lnqZ_arr = np.append(E_lnqZ_arr , E_lnqZ)
          E_lnpZ_arr = np.append(E_lnpZ_arr , E_lnpZ)
          E_lnqLambda_arr = np.append(E_lnqLambda_arr , E_lnqLambda)
          E_lnpLambda_arr = np.append(E_lnpLambda_arr , E_lnpLambda)
          E_Lambda_arr = np.append(E_Lambda_arr , E_Lambda)
          
          
          # Check convergence
          if n > 1:
              converged = ( (ELBO[n] - ELBO[n-1]) < tol)
              print("Converged: ", converged)
         
          n += 1
               
          
         
             
      Sigma = np.linalg.inv(v*W_k)
      
      hdf5_store.close()
      os.remove("cache.hdf5")
      if (showplot == True and D == 2):
          N_points = 200
          P = np.zeros(np.array([N_points,N_points]))
          
          side1 = np.linspace(min(X[:,0]),max(X[:,0]),N_points)
          side2 = np.linspace(min(X[:,1]),max(X[:,1]),N_points)
          Y1,Y2 = np.meshgrid(side1,side2)
          points = np.stack((Y1,Y2),axis = -1)
          
          for i in range(N):
              P +=  multivariate_normal.pdf(points, X[i,:], Sigma)
          
          plt.pcolormesh(Y1,Y2,P)
          plt.scatter(X[:,0],X[:,1])
          plt.show()
      
      elif(showplot == True and D != 2):
          print("The specified data has either 1 or more than 2 variables, so couldn´t plot the distribution")
          
      plt.figure()
      plt.plot(np.arange(len(ELBO)),ELBO,'-')
      
      plt.figure()
      plt.plot(E_lnpX_arr)
      plt.title('E_lnpX_arr')
      
      plt.figure()
      plt.plot(E_lnqZ_arr)
      plt.title('E_lnqZ_arr')
      
      plt.figure()
      plt.plot(E_lnpZ_arr)
      plt.title('E_lnpZ_arr')
      
      plt.figure()
      plt.plot(E_lnqLambda_arr)
      plt.title('E_lnqLambda_arr')
      
      plt.figure()
      plt.plot(E_lnpLambda_arr)
      plt.title('E_lnpLambda_arr')
      
      plt.figure()
      plt.plot(E_Lambda_arr)
      plt.title('E_Lambda_arr')
      
      Return_object.append(Sigma)
      
    return(Sigma, 0 ,0)
    
    
