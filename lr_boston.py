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


result = []
# which feature
for i_feature in range(0,13):
	# Use only one feature
	diabetes_X = boston.data[:, np.newaxis, i_feature]

	# Split the data into training/testing sets
	boston_X_train = diabetes_X[:-20]
	boston_X_test = diabetes_X[-20:]

	# Create linear regression object
	model = linear_model.LinearRegression()

	# Train the model using the training sets
	model.fit(boston_X_train, boston_y_train)
	print "y = %dx + %d" %(model.coef_, model.intercept_)
	# Explained variance score: score=1 is perfect prediction
	score = model.score(boston_X_test, boston_y_test)
	
	result.append(score)

maxindex =  np.argmax(result)
print "Best fitted feature name is: %s" % boston.feature_names[maxindex]
print "Best fitted model score is: %f" % result[maxindex]


plt.plot(boston.data[:, np.newaxis, maxindex][-20:], boston_y_test, 'bo')
plt.show()