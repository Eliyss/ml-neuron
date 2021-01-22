from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import model

(trainX, trainy), (testX, testy) = mnist.load_data()

print(trainX.shape, trainy.shape)
print(testX.shape, testy.shape)

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))

plt.show()


trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
trainY = tf.keras.utils.to_categorical(trainy)
testY = tf.keras.utils.to_categorical(testy)

trainaX = ax.reshape((ax.shape[0], 28, 28, 1))
testaX = tx.reshape((tx.shape[0], 28, 28, 1))
trainaY = np.reshape(ay, (-1, 1))
testaY = np.reshape(ty, (-1, 1))

train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0

testmodel = model(loss_function = 'cross entropy')
testmodel.add(784, activation='reLu')
testmodel.add(16, activation='reLu')
testmodel.add(16, activation='reLu')
testmodel.add(10, activation='softmax')

testmodel.fit(train_norm, trainY, batch_size=100, epoch=100)

print('Accuracy', testmodel.cat_score(test_norm, testY))