# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:36:40 2019

@author: Mads
"""

from vargmm import VB_inference
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from EM_KernelDensity_mads import EM_KernelDensity
import seaborn as sns
from ast import literal_eval
import pickle
import pandas as pd


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
        pca = PCA(n_components=2, whiten=True)
        X = pca.fit_transform(X)
        print("Explained variance ratio for two first components:", pca.explained_variance_ratio_)

    # hold-out
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    class_datasets = [X[y == i] for i in classes]
    return X, y, classes



data = None
X, y, classes = preprocess_data_pure(data, synset='mnist', pca=False)

class_datasets = [X[y == i] for i in classes]

ELBO_list = []
i = 0
for dataset in class_datasets:
    i  = i+1
    print(i)
    sigma, iteration, ELBO_arr = VB_inference(dataset, 'Kfold')
    ELBO_list.append(ELBO_arr)


c1 = np.load('c1.npy')
c3 = np.load('c3.npy')
c8 = np.load('c8.npy')

sns.set()
sns.set_style("whitegrid")
plt.plot(c1 , label = "1")
plt.plot(c3,  label = "3")
plt.plot(c8,  label = "8")
plt.xlabel('Iterations' , fontsize = 16)
plt.ylabel('ELBO')
leg = plt.legend(title=r'$Digits$ : ', ncol=1)
leg._legend_box.align = 'right'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.xlabel(r'\textbf{time} (s)')
plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
plt.title(r"\TeX\ is Number "r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",fontsize=16, color='gray')
# Make room for the ridiculously large title.
plt.subplots_adjust(top=0.8)

plt.savefig('tex_demo')
plt.show()


f = open('elbo.txt', 'r')
ELBO_list = f.readlines()
f.close()


with open('elbo.txt', 'w') as f:
    for item in ELBO_list:
        f.write("%s\n" % item)
        
        
with open("elbo3.txt", "w") as f:
    for s in ELBO_list:
        f.write(str(s) +"\n")
        
      
yo = []
with open("elbo3.txt", "r") as f:
  for line in f:
    yo.append(float(line.strip()))
        
    


##Create a plot with a tex title
ax = plt.subplot(211)
plt.plot(c1,linewidth=2, label = r'$1$')
plt.plot(c3,linewidth=2, label = r'$3$')
plt.plot(c8,linewidth=2, label = r'$8$')

ax = plt.gca()  # or any other way to get an axis object
ax.plot(c3, label=r'$\sin (x)$')
ax.legend()
leg = plt.legend(title=r'$The_{\mathrm{legend}}$ : ', ncol=1)

plt.xlabel(r'$D \:  \mathrm{(mm)}$')
plt.ylabel(r'$\mathrm{Arbitrary \: unit}$')






# Example data
t = np.arange(0.0, 1.0 + 0.01, 0.01)
s = np.cos(4 * np.pi * t) + 2

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(t, s)

plt.xlabel(r'\textbf{time} (s)')
plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
plt.title(r"\TeX\ is Number ",fontsize=16, color='gray')
# Make room for the ridiculously large title.
plt.subplots_adjust(top=0.8)

plt.savefig('tex_demo')
plt.show()










#!python numbers=disable

import pylab
from pylab import arange,pi,sin,cos,sqrt
fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pylab.rcParams.update(params)
# Generate data
x = pylab.arange(-2*pi,2*pi,0.01)
y1 = sin(x)
y2 = cos(x)
# Plot data
pylab.figure(1)
pylab.clf()
pylab.axes([0.125,0.2,0.95-0.125,0.95-0.2])
pylab.plot(x,y1,'g:',label='$\sin(x)$')
pylab.plot(x,y2,'-b',label='$\cos(x)$')
pylab.xlabel('$x$ (radians)')
pylab.ylabel('$y$')
pylab.legend()

