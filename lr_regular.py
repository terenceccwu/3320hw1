import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7.5], [9.1], [13.2], [17.5], [19.3]]

X_test = [[6], [8], [11], [16]]
y_test = [[8.3], [12.5], [15.4], [18.6]]

# 3.3.1 Linear
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_score = lr_model.score(X_test, y_test)
print "Linear regression (order 1) model score is: %.3f" %lr_score

xx = np.linspace(0, 26, 100)
yy = lr_model.predict(xx.reshape(-1, 1))
plt.plot(xx, yy, 'b-')
plt.plot(X_test,y_test,'ro')
plt.suptitle('Linear regression (order 1) result.')
print "y = %.2f + %.2fx" %(lr_model.intercept_, lr_model.coef_)
plt.show()

# 3.3.2 Poly (order 5)
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lr_model.fit(X_train_poly,y_train)

xx_poly = poly.transform(xx.reshape(xx.shape[0], 1))
yy_poly = lr_model.predict(xx_poly)
plt.plot(xx,yy_poly,'b-')
plt.suptitle("Linear regression (order 5) result")

print lr_model.coef_
print lr_model.intercept_

print "Linear regression (order 5) score is: %.3f" %lr_model.score(X_test_poly, y_test)

plt.plot(X_test,y_test,'ro')
plt.show()

# 3.3.3

ridge_model = Ridge(alpha=1, normalize=False)
ridge_model.fit(X_train_poly, y_train)
yy_ridge = ridge_model.predict(xx_poly)

print ridge_model.coef_
print ridge_model.intercept_

print "Ridge regression (order 5) score is: %.3f" %ridge_model.score(X_test_poly, y_test)

plt.plot(xx,yy_ridge,'b-')
plt.plot(X_test,y_test,'ro')
plt.suptitle("Ridge regression (order 5) result")
plt.show()