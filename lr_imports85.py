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
dropdf = df.dropna()

# 3.2.3
slice = int(round(len(dropdf)*0.8))
train_x = dropdf[:slice]['engine-size'].values.reshape(-1,1)
train_y = dropdf[:slice]['price'].values.reshape(-1,1)
test_x = dropdf[slice:]['engine-size'].values.reshape(-1,1)
test_y = dropdf[slice:]['price'].values.reshape(-1,1)

model = linear_model.LinearRegression()
model.fit(train_x,train_y)

plt.scatter(test_x, test_y, c='b')
plt.plot(test_x, model.predict(test_x), 'r-')
plt.xlabel("Engine size")
plt.ylabel("Price")
plt.show()

print "Price prediction for engine size equals to 175 is: %.2f" % model.predict(175)

# 3.2.4

scaler = StandardScaler()

scaled_train_x = scaler.fit_transform(train_x.astype('float'))
scaled_train_y = scaler.fit_transform(train_y.astype('float'))
scaled_test_x = scaler.fit_transform(test_x.astype('float'))
scaled_test_y = scaler.fit_transform(test_y.astype('float'))

model.fit(scaled_train_x,scaled_train_y)

plt.scatter(scaled_test_x, scaled_test_y, c='b')
plt.plot(scaled_test_x, model.predict(scaled_test_x), 'r-')
plt.xlabel("Standard Engine size")
plt.ylabel("Standard Price")
plt.show()