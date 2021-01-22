from sklearn.utils import shuffle
from layers import Layer
from layers import InputLayer
import numpy as np 

class model:
    def __init__(self, loss_function = 'mean squared'):
        self.layers = []
        self.shape = []
        self.loss_func = loss_function
    
    def add(self, num_nodes, layer_type=None, activation='sigmoid'):
        if self.layers:
            self.layers.append(Layer(self.shape[-1], num_nodes, activation))
            self.shape.append(num_nodes)
        else:
            self.layers.append(InputLayer(num_nodes))
            self.shape.append(num_nodes)
    
    def fit(self, X, y, lr=0.1, batch_size=32, epoch=100):

        size = len(X)
        for ii in range(epoch):
            X_shuffle, Y_shuffle = shuffle(X, y)
            index = batch_size
            while index <= size:
                Xbatch = X_shuffle[index-batch_size:index]
                Ybatch = Y_shuffle[index-batch_size:index]
                gradient, bias_gradient, loss = self.batch_grad(Xbatch, Ybatch, batch_size)
                
                for i in range(1, len(self.layers)):
                    self.layers[i].update(gradient[i], bias_gradient[i], lr)
                    
                print('epoch', ii, 'batch', index/batch_size, 'loss', loss)
                index+=batch_size
        
    def cost(self, pred, y):
        loss = 0
        m = y.shape[0]
        if self.loss_func == 'mean squared':
            mean_squared_error = np.square(np.subtract(y, pred))
            loss = np.sum(mean_squared_error)/m
        elif self.loss_func == 'cross entropy':
            log_likelihood = -np.multiply(y, np.log(np.add(1e-15, pred)))
            loss = np.sum(log_likelihood)/m
        return loss

    def dcost(self, pred, y):
        if self.loss_func == 'mean squared':
            cost = np.subtract(y, pred)
            d_cost = 2*cost
        elif self.loss_func == 'cross entropy':
            m = y.shape[0]
            cost = -np.divide(y, pred)
            d_cost = pred/m
        return d_cost
       
    def batch_grad(self, X, y, batch_size):
        all_grad = []
        all_bias = []
        all_loss = []
        for i in range(batch_size):
            sampleX = np.reshape(X[i], (-1, 1))
            sampleY = np.reshape(y[i], (-1, 1))
            
            activation = [0 for i in range(len(self.shape))]
            error = [0 for i in range(len(self.shape))]
            grad = [0 for i in range(len(self.shape))]

            activation[0] = sampleX

            for i in range(1, len(self.layers)):
                activation[i] = self.layers[i].predict(activation[i-1])

            #print('pred', activation[-1])
            #print('y', sampleY)
            loss = self.cost(activation[-1], sampleY)
            dcost = self.dcost(activation[-1], sampleY)
            #print('dcot', dcost)
            all_loss.append(loss)
            error[-1] = np.multiply(dcost, self.layers[-1].backError())
            grad[-1] = np.matmul(error[-1], activation[-2].T)

            for i in range(len(self.layers)-2, 0, -1):
                error[i] = np.multiply(np.matmul(self.layers[i+1].weights.T, error[i+1]), self.layers[i].backError())
                grad[i] = np.matmul(error[i], activation[i-1].T)

            all_grad.append(grad)
            all_bias.append(error)

        return np.mean(all_grad, axis=0), np.mean(all_bias, axis=0), np.mean(all_loss)
        
    def predict(self, X):
        output = np.reshape(X, (-1, 1))

        for i in range(1, len(self.layers)):
            output = self.layers[i].predict(output)
        
        return output
    
    def score(self, X, y):
        correct = 0

        for i in range(X.shape[0]):
            output = self.predict(X[i])

            if y[i] == (lambda x: 1.0*(x >= 0.5))(output):
                correct += 1

        return correct/X.shape[0]
    
    def cat_score(self, X, y):
        correct = 0

        for i in range(X.shape[0]):
            output = self.predict(X[i])
            choice = np.max(output)
            output = np.array([(lambda x: 1.0*(x == choice))(i) for i in output])

            if np.matmul(y[i], output):
                correct += 1
        
        return correct/X.shape[0]