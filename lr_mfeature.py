import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
import seaborn
seaborn.set()

# 3.2.1
names = ['symboling', 'normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']

df = pd.read_csv('imports-85.data',
                 header=None,
                 names=names,
                     na_values=('?'))

# 3.2.2
scaler = StandardScaler()

dropdf = df.dropna()
standard_x = scaler.fit_transform(dropdf[['engine-size','peak-rpm']].values.astype('float64'))
x = np.append(np.ones((159,1)), standard_x, axis=1)
y = scaler.fit_transform(dropdf['price'].values.reshape(-1,1).astype('float64'))

x_tran = np.transpose(x)
theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_tran, x)), x_tran), y)
print "Parameter theta calculated by normal equation: %.3f, %.3f, %.3f" % (theta[0,0], theta[1,0], theta[2,0])

sgd = linear_model.SGDRegressor(loss='squared_loss')
sgd.fit(standard_x,y.reshape(-1))

print "Parameter theta calculated by SGD: %.3f, %.3f, %.3f" %(sgd.intercept_, sgd.coef_[0], sgd.coef_[1])