This repository contains 2 Python script ONE IS scikit_learn_ml_algorithm_implementation and second is preprocessing.

## The scikit_learn_ml_algorithm_implementation script is that demonstrate the use of various machine learning algorithms and techniques from the Scikit-learn library.

First, the script imports the Iris dataset using Scikit-learn's built-in dataset library and performs exploratory data analysis (EDA) on the data. The script also pre-processes the data by splitting it into training and testing sets, and then trains several machine learning models on the data, including linear regression, multiple linear regression, and logistic regression.

For each model, the script prints the model's coefficients and intercept, as well as predictions and evaluation metrics such as mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE).

Additionally, the script includes functions that calculate predictions using the linear regression model, as well as the logistic regression model. Finally, the script prints a confusion matrix and classification report for the logistic regression model.

Overall, this script serves as a useful reference for data analysts and machine learning practitioners who want to learn more about implementing different machine learning algorithms using Scikit-learn in Python.

This repository contains a Python script for  preprocessing tasks using scikit-learn library. The script performs the following tasks:

# Handling Missing Values
The script reads in a dataset with missing values and handles the missing values using SimpleImputer from scikit-learn library.

# Handling Categorical Values
The script reads in a dataset with categorical features and handles the categorical features using OrdinalEncoder and OneHotEncoder from scikit-learn library.

# Discretization
The script reads in a dataset and performs binning or discretization using KBinsDiscretizer from scikit-learn library.

# Binarization
The script reads in a dataset and performs binarization using Binarizer from scikit-learn library.

# Scaling
The script reads in a dataset and performs different scaling techniques including StandardScaler, MinMaxScaler, MaxAbsScaler and RobustScaler from scikit-learn library.

# Normalization
The script reads in a dataset and performs different normalization techniques including max-norm and l1-norm using normalize from scikit-learn library.

# Requirements
The script requires the following packages to be installed:

numpy
pandas
scikit-learn
