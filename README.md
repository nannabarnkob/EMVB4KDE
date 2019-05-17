# EMVB4KDE

This respository contains code for Expectation-Maximization and Variational Bayes for Kernel Density Estimation. 

The usage of the functions is given here: 


EM_KernelDensity
```
Input:
	Required:
	X (numpy array): An N times d dimensional array containing the data for KDE.
	cross_method (string): Either "Kfold", "HoldOut" or "LeaveOneOut" specifiying the validation metthod.

	Optional:
	MaxIterations (integer): THe maximum number of updates before the function returns.
	tol (float): The tolerance controlling how little the likelihood should change before the function return.
    	Kfolds (inter): The number of folds to use, if the method "Kfold" is used

Output:
    Sigma: The estimated covariance matrix
    Iteration: The number of iterations used
    likelihood_arr: An array containg the likelihood of the model in each iteration.
```

EM_KernelDensity
```
Input:
	Required:
	X (numpy array): An N times d dimensional array containing the data for KDE.
	cross_method (string): Either "Kfold", "HoldOut" or "LeaveOneOut" specifiying the validation metthod.

	Optional:
	MaxIterations (integer): THe maximum number of updates before the function returns.
	tol (float): The tolerance controlling how little the likelihood should change before the function return.
	Kfolds (inter): The number of folds to use, if the method "Kfold" is used

Output:
    Sigma: The estimated covariance matrix
    Iteration: The number of iterations used
    ELBO: An array containg the Evidence Lower Bound of the model in each iteration.

```
classify_EM
```
Input:
	Required:

    Optional:
	data: The dataset to classify
        cross_method = (string): Either "Kfold", "HoldOut" or "LeaveOneOut" specifiying the validation metthod.
	plot (boolean): Whether to plot the classification with the data
        data_augmentation (boolean): Whether to generate more data via the estimated density


Output:
```
ErrorEst (in TwoFoldCrossValidation.py)
```
Input:

    Required:
	None 
	
    Optional:
	data: The dataset to estimate model error on
	cross_method = (string): Either "Kfold", "HoldOut" or "LeaveOneOut" specifiying the validation metthod.
        plot (boolean): Whether to plot the classification with the data
        data_augmentation (boolean): Whether to generate more data via the estimated density
        N_aug_sampes (integer): How many samples to generates
        Outer_K (integer): The number of folds in the outer validation loop
        estimator (string): Which method to use to estimate covariance matrix. Either "EM_KernelDensity" or "VB_KernelDensity"

Output:
    Sigma: The estimated covariance matrix
    Iteration: The number of iterations used
    ELBO: An array containg the Evidence Lower Bound of the model in each iteration.
```
