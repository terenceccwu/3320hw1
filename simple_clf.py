import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split


n_samples = 5000

centers = [(-2, -2), (2, 2)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=42)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)

# 4.1
log_reg = linear_model.LogisticRegression()
log_reg.fit(X_train,y_train)

y_predict = log_reg.predict(X_test)

for i in range(0,len(X_test)):
	if y_predict[i]:
		plt.scatter(X_test[i,0],X_test[i,1], c='r')
	else:
		plt.scatter(X_test[i,0],X_test[i,1], c='b')

plt.savefig('4.1.png')
plt.clf()

# 4.2
counter = 0
for i in range(0,len(y_test)):
	if y_test[i] != y_predict[i]:
		counter += 1

print "Number of wrong predictions is: %d" %counter