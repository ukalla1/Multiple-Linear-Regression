import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder		# To one hot encode the categorical data
from sklearn.compose import ColumnTransformer						# To one hot encode only one column in the data set
from sklearn.impute import SimpleImputer							# to convert any nan values into a numerical value
from sklearn.model_selection import train_test_split				# to break the dataset into train and test
import statsmodels.regression.linear_model as sm 					# to perform backward elimination
from sklearn.linear_model import LinearRegression

def BackwardElimination(x, y, sl):
	var_len = len(x[0])
	for i in range(0, var_len):
		regressor_OLS = sm.OLS(endog = y, exog = x).fit()
		maxVar = max(regressor_OLS.pvalues)
		if maxVar >= sl:
			for j in range(0, var_len - i):
				if (regressor_OLS.pvalues[j].astype(float) == maxVar):
					x = np.delete(x, j, 1)
	print(regressor_OLS.summary())
	return x

# Read the csv into data variable
data = pd.read_csv('./50_Startups.csv')

#Breaking the data into dependent(x) and independent(y) vars
x = data.iloc[:, :-1]
y = data.iloc[:, 4:]

# one hot encoding the categorical data
column_transformer = ColumnTransformer([("State", OneHotEncoder(categories='auto'), [3])], remainder = 'passthrough')
x = column_transformer.fit_transform(x)

# to account for dummy variable trap
x = x[:, 1:]

# appending a column of ones to include the consts
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)

#Performing backward elimination and picking only imprtant features
sl = float(5/100)
x_opt = x[:, [0,1,2,3,4]]
x_ret = BackwardElimination(x_opt, y, sl)
print("x:{}\tx_opt{}".format(x_opt.shape,x_ret.shape))

# splitting the data into train and test data sets
x_train, x_test, y_train, y_test = train_test_split(x_ret, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# fitting the regressor with the r=training data
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting using the regressor with test data
y_pred = regressor.predict(x_test)

# printing the predictions and the test values
print(y_pred, y_test)