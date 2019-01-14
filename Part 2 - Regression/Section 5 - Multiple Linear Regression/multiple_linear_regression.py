# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:] # only incluiding n-1 dummy variables

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # in this LinearRegression object is take into account the implicit x0 ind. variable from the 
                               # y = b0*x0 + b1*x1 + b2 * x2 + ... + bn * xn equation
regressor.fit(X_train, y_train)
# We are not plotting any graph because we have multiple ind. var. so it would be a 5th dim. graph
# and we couln't aprecciate it well. 5 dimensions because 4 ind. var. + 1 dep. var.

# Predicting the Test set results with the help of the LinearRegression obj
y_pred = regressor.predict(X_test)
# Until this point we used all the ind. var.

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm  
# in this library is not take into account the implicit x0 ind. var. We don't have what's called "intercept"
# so we need to add a column of 1's to compensate that (x0 always has a value of 1)

# We're doing this because we want that the column of 1's is at the beginning
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)  # axis = 1 because we're adding a columns

#       Performing backward elimination method technique with a S.L. (significance level) = 0.005 or 5%
X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]    # X_opt is a matrix that will only contain the relevant ind. var. that have high impact for the dep. var.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #OLS (ordinary least squares)
regressor_OLS.summary()

# Eliminating the ind. var. with the highest p-value
X_opt = X_train[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_train[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_train[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_train[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_test = np.append(arr = np.ones((10, 1)).astype(int), values = X_test, axis = 1) # only 10 rows of 1's because the test set is 20% of the actual dataset size
X_test = X_test[:, [0,3]] # just the 1rst and 4th colums because they correspond to the 1's and R&D features that are present in our final X_opt equation
Y_pred_ols = regressor_OLS.predict(X_test)








