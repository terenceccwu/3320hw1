#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn; seaborn.set()

# Load the diabetes dataset
boston = datasets.load_boston()

# Split the targets into training/testing sets
boston_y_train = boston.target[:-20]
boston_y_test = boston.target[-20:]

def regression(i_feature):
	# Use only one feature
	diabetes_X = boston.data[:, np.newaxis, i_feature]

	# Split the data into training/testing sets
	boston_x_train = diabetes_X[:-20]
	boston_x_test = diabetes_X[-20:]

	# Create linear regression object
	model = linear_model.LinearRegression()

	# Train the model using the training sets
	model.fit(boston_x_train, boston_y_train)
	# print "y = %.2fx + %.2f" %(model.coef_, model.intercept_)
	# Explained variance score: score=1 is perfect prediction
	score = model.score(boston_x_test, boston_y_test)

	return {'model': model, 'score': score}

# 3.1.2 Find best fitted feature

result = []

for i_feature in range(0,13):
	result.append(regression(i_feature)['score'])

bestfit_feature =  np.argmax(result)
bestfit_name = boston.feature_names[bestfit_feature]
print "Best fitted feature name is: %s" % bestfit_name
print "Best fitted model score is: %f" % result[bestfit_feature]

# 3.1.3 Calculate the loss function
model = regression(bestfit_feature)['model']
bestfit_test = boston.data[:, np.newaxis, bestfit_feature][-20:]
predict_y = model.predict(bestfit_test)
sse = np.mean((predict_y - boston_y_test)**2)
print "Value of the loss function for the best fitted model is: %f" %sse

# 3.1.4 Plot data
print predict_y
plt.scatter(bestfit_test, boston_y_test, c='b')
plt.plot(bestfit_test, predict_y, 'r-')
plt.xlabel(bestfit_name)
plt.ylabel("Boston House Price")
plt.show()

