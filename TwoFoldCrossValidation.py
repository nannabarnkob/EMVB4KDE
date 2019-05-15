# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:32:37 2019

@author: Mads
"""


from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from EM_KernelDensity import EM_KernelDensity
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from VB_KernelDensity import VB_inference

sns.set()

def ErrorEst(data=None, cross_method='Kfold', plot=False,data_augmentation=True , N_aug_samples=100 ,  outerK = 10 , estimator = "EM_KernelDensity"):
    np.random.seed(1234)
    
    if estimator == "EM_KernelDensity":
        KDE_estimator = EM_KernelDensity
    elif estimator == "VB_inference":
        KDE_estimator = VB_inference
    elif estimator == "Scipy":
        KDE_estimator = ScipyKDE
    #KDE_estimatorSlow = VB_inference_Slow
    
    KDE_estimator = VB_inference
    
    X, y,classes = preprocess_data_pure(data, synset='mnist', pca=True)  
    error_arr = []
    error_arr_upsample = []
    kf = KFold(n_splits = outerK,shuffle=False)
    for k , (train_index, test_index) in enumerate(kf.split(X)):
        print(k)

        X_train, X_test = X[train_index], X[test_index] , 
        y_train , y_test = y[train_index], y[test_index]
        class_datasets = [X_train[y_train == i] for i in classes]

        Sigmas = []
        Sigmas_upsample = []
        for dataset in class_datasets:
            sigma, iteration, likelihood_arr = KDE_estimator(dataset,cross_method)
            Sigmas.append(sigma)

  
        error, accuracy, y_hat = get_performance_pure(X_test, y_test, class_datasets, classes, Sigmas)
        print(error)
        error_arr = np.append(error_arr , error)   

            
        if data_augmentation:
            for i in range(len(class_datasets)):
                Xsample = sample(Sigmas[i], class_datasets[i], N_aug_samples)
                class_datasets[i] = np.vstack((class_datasets[i], Xsample))
                
            for dataset in class_datasets:
                sigma, iteration, likelihood_arr = KDE_estimator(dataset, cross_method)
                Sigmas_upsample.append(sigma)
                
            error, accuracy, y_hat = get_performance_pure(X_test, y_test, class_datasets, classes, Sigmas_upsample)
            error_arr_upsample = np.append(error_arr_upsample , error)   
        
        
    if data_augmentation:
        return(np.vstack( (np.mean(error_arr) , np.mean(error_arr_upsample))))
        
    else:
        return( np.mean(error_arr) )

def get_synthetic_data():
    X0 = np.random.multivariate_normal([0, 0], cov=[[1, -2], [3, -4]], size=100)
    X1 = np.random.multivariate_normal([0, 0], cov=[[1, 2], [3, 4]], size=100)
    y0 = np.zeros(len(X0))
    y1 = np.ones(len(X1))
    np.vstack((y0, y1)).flatten()
    fig, ax = plt.subplots()
    ax.scatter(X0[:, 0], X1[:, 1], marker='o')
    ax.scatter(X0[:, 0], X1[:, 1], marker='o')
    plt.show()
    return X0, X1


def plot_density(points, sigma):
    # Get density estimation for grid for plotting
    grid = 200
    phi_m = np.linspace(-3, 3, grid)
    phi_p = np.linspace(-3, 3, grid)
    X, Y = np.meshgrid(phi_m, phi_p)
    dens = np.zeros((grid, grid))
    for i in range(grid):
        for j in range(grid):
            dens[i, j] += np.sum(multivariate_normal.pdf(points, np.array([X[i, j], Y[i, j]]), sigma))
    return dens


def preprocess_data_pure(data=None, synset=None, pca=True):
    # specific for income
    # split data in test-train and
    if data:
        data = pd.read_csv(data)
        target = ' income'
        X = data.drop(columns=[target]).values
        y = data.loc[:, target].values
    elif synset == 'iris':
        iris = datasets.load_iris()
        X = iris['data']
        y = iris['target']
    elif synset == 'mnist':
        mnist = datasets.load_digits()
        X = mnist['data']
        X = X + np.random.normal(0, 0.01, size=X.shape)   # avoid zeros
        y = mnist['target']

    classes = np.sort(np.unique(y))

    if pca:
        pca = PCA(n_components=4, whiten=True)
        X = pca.fit_transform(X)
        print("Explained variance ratio for two first components:", pca.explained_variance_ratio_)

    # hold-out
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    class_datasets = [X[y == i] for i in classes]
    return X, y, classes


def classify_cv(data=None):
    class_datasets, X_test, y_test, classes = preprocess_data(data)
    # use grid search cross-validation to optimize the bandwidth
    dim = 2
    Sigmas = []
    params = {'bandwidth': np.logspace(-1, 1, 30)}
    for i in range(len(class_datasets)):
        grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False)
        grid.fit(class_datasets[i])
        print("Best bandwidth for class {0}: {1}".format(classes[i], grid.best_estimator_.bandwidth))
        Sigma = (grid.best_estimator_.bandwidth**2)*np.identity(dim)
        Sigmas.append(Sigma)
    error, accuracy, y_hat = get_performance(X_test, y_test, class_datasets, classes, Sigmas)


def sample(Sigma, dataset, samples=100):
    Xs = []
    for i in range(samples):
        n = np.random.randint(0, len(dataset))
        Xs.append(np.random.multivariate_normal(dataset[n], Sigma))
    Xs = np.array(Xs)
    return Xs




def get_performance_pure(X_test, y_test, class_datasets, classes, Sigmas):
    densities = np.zeros([len(X_test), len(classes)])
    N_classes = [len(X) for X in class_datasets]
    N = np.sum(N_classes)

    for i, n in enumerate(X_test):
        for c in classes:
            densities[i, c] = 1/len(class_datasets[c])*np.sum(multivariate_normal.pdf(class_datasets[c], mean=n, cov=Sigmas[c]))*N_classes[c]/N
    # calculate probabilities
    probabilities = np.divide(densities, np.sum(densities,axis=1).reshape(len(X_test), 1))
    y_hat = np.argmax(probabilities, axis=1)
    error = 1/len(y_test)*np.sum(y_hat != y_test)
    accuracy = 1-1/len(y_test)*np.sum(y_hat != y_test)
    return error, accuracy, y_hat

def ScipyKDE(data=None):
    class_datasets, X_test, y_test, classes = preprocess_data(data)
    # use grid search cross-validation to optimize the bandwidth
    dim = 2
    Sigmas = []
    params = {'bandwidth': np.logspace(-1, 1, 30)}
    for i in range(len(class_datasets)):
        grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False)
        grid.fit(class_datasets[i])
        print("Best bandwidth for class {0}: {1}".format(classes[i], grid.best_estimator_.bandwidth))
        Sigma = (grid.best_estimator_.bandwidth**2)*np.identity(dim)
        Sigmas.append(Sigma)
    error, accuracy, y_hat = get_performance(X_test, y_test, class_datasets, classes, Sigmas)




#classify_EM('../data/cleaned_income_data.csv', cross_method='Kfold', plot=True)
#classify_EM(upsample=True, data_augmentation=True)
#classify_cv()