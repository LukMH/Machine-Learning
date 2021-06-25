import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from keras.datasets import mnist
import torch.optim as optim

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# parameters
learningRate = 10**(-4)
miniBatchSize = 50
inputSize = 784
numClasses = 4
trainSize = np.shape(y_train)[0]

# Prepare the dataset
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (60000, 784)) # Each image is reshaped to a row vector of size 784
x_test = np.reshape(x_test, (10000, 784)) # Each image is reshaped to a row vector of size 784

def binary_encoding(data, size):
  temp = np.zeros((size,4), dtype=int)
  for i in range(size):
    if data[i]==0:
      temp[i,:]=[0,0,0,0]
    elif data[i]==1:
      temp[i,:]=[0,0,0,1]
    elif data[i]==2:
      temp[i,:]=[0,0,1,0]
    elif data[i]==3:
      temp[i,:]=[0,0,1,1]
    elif data[i]==4:
      temp[i,:]=[0,1,0,0]
    elif data[i]==5:
      temp[i,:]=[0,1,0,1]
    elif data[i]==6:
      temp[i,:]=[0,1,1,0]
    elif data[i]==7:
      temp[i,:]=[0,1,1,1]
    elif data[i]==8:
      temp[i,:]=[1,0,0,0]
    elif data[i]==9:
      temp[i,:]=[1,0,0,1]
    else:
      print("error")
  return temp

y_train = binary_encoding(y_train, len(y_train))
y_test = binary_encoding(y_test, len(y_test))

# Convert and load data
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)
train_loader = DataLoader(dataset = train, batch_size = miniBatchSize, shuffle = True)
test_loader = DataLoader(dataset = test, batch_size = 10000, shuffle = True)
input_shape = (-1,inputSize)

# Define last layer
add_layer = np.array([[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[0,0,0,0]])
add_layer = torch.from_numpy(add_layer).type(torch.FloatTensor)

# Class MLP
class MLP(nn.Module):
  def __init__(self, input_size = inputSize, num_classes = numClasses):
    super(MLP, self).__init__()
    # First fully connected layer
    self.fc1 = nn.Linear(input_size, 15)
    # Second fully connected layer
    self.fc2 = nn.Linear(15, 10)
   
  def forward(self,x):
    out = F.relu(self.fc1(x))
    out = F.softmax(self.fc2(out), dim = 1)
    out = torch.matmul(out, add_layer)
    return out

model = MLP()
# Loss Function
criterion = nn.MSELoss()
# Optimizer (AdamAdamOptimizer, learning rate is 1e-4)
optimizer = optim.Adam(model.parameters(), lr = learningRate)
# Define the accuracy
def accuracy(y_true,y_pred):
  prediction = y_pred > 0.5
  prediction = prediction.float()
  temp = prediction == y_true
  temp = temp.int()
  temp = torch.sum(temp,1) == 4
  temp = temp.float()
  output = float(sum(temp)/len(y_true))
  return output

# Train the model
for epoch in range(5):
  for iteration_num, (data, labels) in enumerate(train_loader):
    data = Variable(data.view(input_shape))
    labels = Variable(labels)
    optimizer.zero_grad()
    scores = model(data)
    loss = criterion(scores, labels)
    loss.backward()
    optimizer.step()
    if iteration_num % 300 == 299:    # Print every 300 iterations
      training_accuracy = accuracy(y_true = labels, y_pred = scores) # Calculate the accuracy of the current iteration
      print(f"Training accuracy (Iteration {int(trainSize/miniBatchSize) * epoch + iteration_num + 1}): {training_accuracy:.4f}")
print('Finished Training')


# Calculate the testing accuracy 
with torch.no_grad():
  for (data, labels) in test_loader:
    data = Variable(data.view(input_shape))
    outputs = model(data)
    test_accuracy = accuracy(y_true = labels, y_pred = outputs)
print(f"Test accuracy: {test_accuracy}")
