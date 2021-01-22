import numpy as np
from sklearn import datasets
from model import model
from sklearn.utils import shuffle

iris_data = datasets.load_iris()

print(iris_data.data.shape)

x = iris_data.data
y = iris_data.target

x_shuffle, y_shuffle = shuffle(x, y)

trainX = x_shuffle[:120]
trainY = y_shuffle[:120]

testX = x_shuffle[120:]
testY = y_shuffle[120:]
print(testY)



trainX_range = np.max(trainX, axis=0)-np.min(trainX, axis=0)
trainX_norm = np.divide((trainX-np.min(trainX, axis=0)), trainX_range)

testX_range = np.max(testX, axis=0)-np.min(testX, axis=0)
testX_norm = np.divide((testX-np.min(testX, axis=0)), testX_range)

aaa = np.reshape(trainY, (-1, 1))

testmodel = model()
testmodel.add(4, activation='sigmoid')
testmodel.add(2, activation='sigmoid')
testmodel.add(2, activation='sigmoid')
testmodel.add(1, activation='sigmoid')

testmodel.fit(trainX_norm, aaa, batch_size=10, epoch=10)