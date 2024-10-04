#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.preprocessing import StandardScaler


# In[2]:


def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoid_derivative(X):
    return X* (1-X)

def relu(X):
    return np.maximum(0, X)

def relu_derivative(X):
    return np.where(X > 0, 1, 0)


def crossentropyloss(y, y_preds):
        y_preds = np.clip(y_preds, 1e-15, 1 - 1e-15)
        loss = -np.mean( y * np.log(y_preds) + (1-y) * np.log(1-y_preds))
        return loss

class NeuralNetwork():
    def __init__(self, input_size: int, hidden_units: int, output_size:int):
        # self.weights_ip = np.round(np.random.rand(input_size, hidden_units),2)  
        # self.bias_ip = np.zeros((hidden_units)) 
        # self.weights_op = np.round(np.random.rand(hidden_units, output_size),2)  
        # self.bias_op = np.zeros((output_size)) 
        self.weights_ip = np.random.randn(input_size, hidden_units) * np.sqrt(2 / (input_size))
        self.bias_ip = np.zeros((1, hidden_units))  # Hidden bias (1, hidden_units)
        self.weights_op = np.random.randn(hidden_units, output_size) * np.sqrt(2 / (hidden_units + output_size))
        self.bias_op = np.zeros((1, output_size))
        

    def __call__(self,X):
        return self.forward(X)

    def forward(self, X):
        self.m = X.shape[0]
        self.X = X
        self.a1 = (self.X @ self.weights_ip) + self.bias_ip
        self.z1 = relu(self.a1)
        self.a2 = (self.z1 @ self.weights_op) + self.bias_op
        self.z2 = sigmoid(self.a2)
        # print("weights= ",self.weights_ip,self.weights_op)
        # print("bias= ",self.bias_ip,self.bias_op)
        # print("a1= ",self.a1)
        # print("z1= ",self.z1)
        # print("a2= ",self.a2)
        # print("z2= ",self.z2)
        return self.z2

    def loss(self,y):
        return crossentropyloss(y,self.z2)

    def backward(self, y, learning_rate):
        dz2 = self.z2 - y        
        dw2 = (1/self.m)*(self.z1.T @ dz2)
        db2 = (1/self.m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = dz2 @ self.weights_op.T
        
        dz1 = da1* relu_derivative(self.z1)
        
        dw1 = (1/self.m)*(self.X.T @ dz1)
        db1 = (1/self.m) * np.sum(dz1, axis=0, keepdims=True) 
        
        # print("dz2= ",dz2)
        # print("dw2=",dw2)
        # print("db2=",db2)
        # print("da1= ",da1)
        # print("dz1= ",dz1)
        # print("dw1=",dw1)
        # print("db1=",db1)
        self.weights_op = self.weights_op - learning_rate*dw2
        self.bias_op = self.bias_op - learning_rate*db2
        self.weights_ip = self.weights_ip - learning_rate*dw1
        self.bias_ip = self.bias_ip - learning_rate*db1




# In[5]:


def custom_train_test_split(X,y,ratio):
    length = X.shape[0]
    indices = np.arange(length)
    np.random.shuffle(indices)
    split_size = int(length - length*ratio)
    train_indices = indices[:split_size]
    test_indices = indices[split_size:]
    return X[train_indices], X[test_indices],y[train_indices], y[test_indices]



# In[6]:


def train_test_data():
    np.random.seed(42)
    num_samples = 400
    heights = np.random.normal(loc=170, scale=10, size=num_samples)  # Mean height ~ 170 cm
    weights = np.random.normal(loc=70, scale=15, size=num_samples)    # Mean weight ~ 70 kg
    bmi = weights / (heights / 100) ** 2  
    y = (bmi < 24.9).astype(int)  
    X = np.column_stack((heights, weights))
    y = y.reshape(-1,1)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y,0.25)
    return X_train, X_test, y_train, y_test 


X_train, X_test, y_train, y_test = train_test_data()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


model = NeuralNetwork(X_train.shape[1],16,1)
def train(X,y):
    epochs = 100
    for i in range(epochs):
        model(X)
        loss = model.loss(y)
        model.backward(y,0.1)
        print(loss)

def test(X,y):
    y_preds = model(X)
    output = np.where(y_preds >= 0.5, 1, 0)
    accuracy = np.mean(output==y)
    return accuracy


# In[8]:


train(X_train,y_train)


# In[9]:


test(X_train,y_train)


# In[10]:


test(X_test,y_test)


# In[ ]:


# num_samples = 100
# test_heights = np.random.normal(loc=170, scale=10, size=num_samples)  # Mean height ~ 170 cm
# test_weights = np.random.normal(loc=70, scale=15, size=num_samples) 
# bmi = weights / (heights / 100) ** 2  
# y_test = (bmi < 24.9).astype(int)  

# X_test = np.column_stack((heights, weights))
# y = y.reshape(-1,1)


# scaler = StandardScaler()
# X = scaler.fit_transform(X)


# In[ ]:




