import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
import math
import time as time
import os
from sklearn.model_selection import KFold , LeaveOneOut , train_test_split
from scipy.spatial.distance import pdist, squareform
from scipy.special import multigammaln
import h5py
from scipy.stats import multivariate_normal


os.chdir("C:/Users/massi/Dropbox/Advanced Machine learning med de seje og Nanna/KDE/Nanna")
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
  
def VB_inference(data,method,MaxIterations=300,showplot = False,Kfolds = 10, tol = 10e-3):    
    #Get number of differnet classes

      Return_object = []

      X = data
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
      

      with h5py.File("cache.hdf5",  "a") as f:
          if f.get('results'):
              os.remove("cache.hdf5")
              f.close()
        #del f["results"]
      #Create cache
      hdf5_store = h5py.File("./cache.hdf5", "a")
      Outer_DM = hdf5_store.create_dataset("results", (D,D,(N)**2), compression="gzip")

      
      #Create a 3d-array with outerproduct of the pairwise differences
     # Outer_DM = np.empty((D,D,(N)**2))
      print("Create a 3d-array with outerproduct of the pairwise differences")
      
      m = 50
      
      start = time.time()
      for h in range(int(np.ceil(N/m))):
        
        Holder = np.empty((D,D,N*m))

        if ((h+1)*m) > N:
          Holder = np.empty((D,D,((N-m*h)*N)))
        else:
          Holder = np.empty((D,D,N*m))
          
   
        M = int(len(Holder[0][0])/N)
        for i in range(M):
          
          for j in range(N):
            #print(j)
            diff = DM[h*m+i,j,:]
            Holder[:,:,i*N+j] = np.outer(diff,diff)
        
        Outer_DM[:,:,np.arange((h*m*N),(h*m*N + M*N))] = Holder

      end = time.time()
      print(end-start)
      
      
      #Check if the method is holdout
      if method == "HoldOut":
          X_train, X_test, train_index, test_index = train_test_split(
          X, range(N), test_size=0.4, random_state=1)
          N_train, D_train = X_train.shape
          N_test, D_test = X_test.shape
      
      if method == "Kfold":
          kf = KFold(n_splits=Kfolds,shuffle=False)
          
      if method == "LeaveOneOut":
          kf = LeaveOneOut()
      
      
      ELBO = np.array([])
      
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
          if method == "LeaveOneOut" or method == "Kfold":
              
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
     
 
        
          elif method == "HoldOut":
              v = v0 + N_test
              
              E_Lambda = sum(digamma((v+1-(np.arange(D)+1))/2))+ D*math.log(2)+logdet(W)
              E_pi = np.log(1/N_train)
              
              for i in range(N_test):
                  ln_rho = E_pi + 1/2*E_Lambda - D/2*math.log(2*math.pi) - \
                  1/2*v*np.sum(np.dot((X_test[i,:] - X_train),W)*(X_test[i,:] - X_train),axis = 1)
                  
                  ln_C = -max(ln_rho)
                  R[test_index[i],train_index] = np.exp(ln_rho+ln_C)/sum(np.exp(ln_rho+ln_C))
          
          else:
              print("Could recognize the specified method. Please check for spelling errors")
              return()
     
         
         
          flat_R = np.reshape(R,np.array([1,N**2]))
          W_inv_new = np.linalg.inv(W0) + np.sum(flat_R*Outer_DM[:,:,:],axis = -1)
          
          #Update W_k
          W_k = np.linalg.inv(W_inv_new)
          
          
          # Compute Evidence Lower Bound
          if method == "LeaveOneOut" or method == "Kfold":
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
            
          elif method == "HoldOut":
              
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
                
          
          E_lnqLambda = (v-D-1)/2*(sum(digamma((v+1-(np.arange(D)+1))/2))+ D*math.log(2)+logdet(W_k)) -1/2*v*D - \
          (v*D/2*math.log(2)+v/2*logdet(W_k)+multigammaln(v/2,D))
          
          E_lnpLambda = (v0-D-1)/2*(sum(digamma((v+1-(np.arange(D)+1))/2))+ D*math.log(2)+logdet(W_k)) - \
          1/2*v*np.matrix.trace(np.dot(np.linalg.inv(W0),W_k)) - \
          (v0*D/2*math.log(2)+v0/2*logdet(W0)+0)
          
          
          # Compute Evidence lower bound
          ELBO = np.append(ELBO,np.array([E_lnpX - E_lnqZ + E_lnpZ - E_lnqLambda + E_lnpLambda]))
          
          # Check convergence
          if n > 1:
              converged = ( (ELBO[n] - ELBO[n-1]) < tol)
              #print("Converged: ", converged)
         
          n += 1
               
          
         
             
      Sigma = np.linalg.inv(v*W_k)
      
      hdf5_store.close()
      end = time.time()
      os.remove("cache.hdf5")
      print(end-start)
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
          
# =============================================================================
#       plt.figure()
#       plt.plot(np.arange(len(ELBO)),ELBO,'-')
#       
#       plt.figure()
#       plt.plot(E_lnpX_arr)
#       plt.title('E_lnpX_arr')
#       
#       plt.figure()
#       plt.plot(E_lnqZ_arr)
#       plt.title('E_lnqZ_arr')
#       
#       plt.figure()
#       plt.plot(E_lnpZ_arr)
#       plt.title('E_lnpZ_arr')
#       
#       plt.figure()
#       plt.plot(E_lnqLambda_arr)
#       plt.title('E_lnqLambda_arr')
#       
#       plt.figure()
#       plt.plot(E_lnpLambda_arr)
#       plt.title('E_lnpLambda_arr')
#       
#       plt.figure()
#       plt.plot(E_Lambda_arr)
#       plt.title('E_Lambda_arr')
# =============================================================================
      
      Return_object.append(Sigma)

      return(Sigma,n,ELBO)
    
    
