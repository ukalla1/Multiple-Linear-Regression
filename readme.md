This project is the implementation of backward elimination and implementation of multiple linear regression in python using the scikit learn library.
The following are the goals of the code:
	1. To read a CSV file with the data inputs.
	2. Store the data into 2 variables based on dependent and independent vars (Here, it is assumed that the data has independent vars in all but the last column, with the data itself having 5 columns).
	3. OneHotEncode the categorical data
	4. Eliminate dummy variable trap
	5. Perform backward elimination to store only the necessary features (Here the significance level is 5% i.e. sl = 0.05)
	6. Fit this data to the regressor
	7. Predict the profits using the test data for this regressor