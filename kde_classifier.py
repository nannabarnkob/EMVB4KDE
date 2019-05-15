
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from EM_KernelDensity import EM_KernelDensity
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns

def classify_EM(data=None, cross_method='Kfold', plot=False, upsample=False, data_augmentation=False):
    class_datasets, X_test, y_test, classes = preprocess_data(data, synset='mnist', pca=False)

    # find covariance matrix for both sets
    Sigmas = []
    for dataset in class_datasets:
        sigma, iteration, likelihood_arr = EM_KernelDensity(dataset, cross_method)
        Sigmas.append(sigma)
    # pdb.set_trace()
    error, accuracy, y_hat = get_performance(X_test, y_test, class_datasets, classes, Sigmas)

    if plot:
        fig, ax = plt.subplots()
        Z = 0
        for i in range(len(class_datasets)):
            Z += plot_density(class_datasets[i], Sigmas[i])
            ax.scatter(class_datasets[i][:, 0], class_datasets[i][:, 1], marker='o', zorder=5)
            ax.scatter(X_test[y_hat == classes[i], [0]], X_test[y_hat == classes[i], [1]], marker='x', zorder=10)
        grid = 200
        phi_m = np.linspace(-3, 3, grid)
        phi_p = np.linspace(-3, 3, grid)
        X, Y = np.meshgrid(phi_m, phi_p)
        # plot density in the background
        ax.pcolor(X, Y, Z, vmin=abs(Z).min(), vmax=abs(Z).max(), zorder=1)
        plt.show()

    # upsample dataset
    if upsample:
        fig, ax = plt.subplots()
        for i in range(len(class_datasets)):
            Xsample = sample(Sigmas[i], class_datasets[i], samples=100)
            ax.scatter(Xsample[:, 0], Xsample[:, 1], marker='x', zorder=10)
            ax.scatter(class_datasets[i][:, 0], class_datasets[i][:, 1], marker='o', zorder=5)
            if data_augmentation:
                class_datasets[i] = np.vstack((class_datasets[i], Xsample))
        plt.show()

        # rerun the whole thing
        Sigmas = []
        for dataset in class_datasets:
            sigma, iteration, likelihood_arr = EM_KernelDensity(dataset, cross_method)
            Sigmas.append(sigma)
        error, accuracy, y_hat = get_performance(X_test, y_test, class_datasets, classes, Sigmas)


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


def preprocess_data(data=None, synset=None, pca=True):
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
        pca = PCA(n_components=2, whiten=True)
        X = pca.fit_transform(X)
        print("Explained variance ratio for two first components:", pca.explained_variance_ratio_)

    # hold-out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    class_datasets = [X_train[y_train == i] for i in classes]
    return class_datasets, X_test, y_test, classes


def classify_cv(data=None):
    class_datasets, X_test, y_test, classes = preprocess_data(data)
    # use grid search cross-validation to optimize the bandwidth
    dim = 64
    Sigmas = []
    params = {'bandwidth': np.logspace(-1, 1, 30)}
    for i in range(len(class_datasets)):
        grid = GridSearchCV(KernelDensity(), params, cv=10, iid=False)
        grid.fit(class_datasets[i])
        print("Best bandwidth for class {0}: {1}".format(classes[i], grid.best_estimator_.bandwidth))
        Sigma = (grid.best_estimator_.bandwidth**2)*np.identity(dim)
        Sigmas.append(Sigma)
    error, accuracy, y_hat = get_performance_pure(X_test, y_test, class_datasets, classes, Sigmas)


def sample(Sigma, dataset, samples=100):
    Xs = []
    for i in range(samples):
        n = np.random.randint(0, len(dataset))
        Xs.append(np.random.multivariate_normal(dataset[n], Sigma))
    Xs = np.array(Xs)
    return Xs


def classify_cv_2level(class_data,crossmethod):
    #class_datasets, X_test, y_test, classes = preprocess_data(data)
    # use grid search cross-validation to optimize the bandwidth
    dim = class_data.shape[1]
    Sigmas = []
    params = {'bandwidth': np.logspace(-1, 1, 30)}
    grid = GridSearchCV(KernelDensity(), params, cv=10, iid=False)
    grid.fit(class_data)
    #print("Best bandwidth for class {0}: {1}".format(classes[i], grid.best_estimator_.bandwidth))
    Sigma = (grid.best_estimator_.bandwidth**2)*np.identity(dim)
    return(Sigma,0,0)


def get_performance(X_test, y_test, class_datasets, classes, Sigmas):
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
    print("Error:", error, "Accuracy:", accuracy)
    return error, accuracy, y_hat


#classify_EM('../data/cleaned_income_data.csv', cross_method='Kfold', plot=True)
#classify_EM(upsample=True, data_augmentation=True)
#classify_cv()