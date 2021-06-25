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

# Randomly visualise 1 image for each class in the training dataset
print(f"Randomly visualise 1 image for each class in the training dataset:")
for i in range(10):
  plt.subplot(2,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  randomNumber = random.randint(0, len(x_train[y_train == i]) - 1)
  plt.imshow(x_train[y_train == i][randomNumber], cmap = 'gray')
  plt.xlabel(f"class {i}")
plt.tight_layout(pad = 0.25)
plt.show()

# Randomly visualise 1 image for each class in the testing dataset
print(f"Randomly visualise 1 image for each class in testing dataset:")
for i in range(10):
  plt.subplot(2,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  randomNumber = random.randint(0, len(x_test[y_test == i]) - 1)
  plt.imshow(x_test[y_test == i][randomNumber], cmap = 'gray')
  plt.xlabel(f"class {i}")
plt.tight_layout(pad = 0.25)
plt.show()

# Prepare the dataset
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (60000, 784)) # Each image is reshaped to a row vector of size 784
x_test = np.reshape(x_test, (10000, 784)) # Each image is reshaped to a row vector of size 784
# Convert and load data
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)
train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)
train_loader = DataLoader(dataset = train, batch_size = 50, shuffle = True)
test_loader = DataLoader(dataset = test, batch_size = 10000, shuffle = True)
input_shape = (-1,784)

# Class CNN
class CNN(nn.Module):
  def __init__(self, in_channels = 1, num_classes = 10):
    super(CNN, self).__init__()
    # First convolutional layer
    self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 25, kernel_size = (12,12), stride = (2,2), padding = (0,0))
    # Second convolutional layer
    self.conv2 = nn.Conv2d(in_channels = 25, out_channels = 64, kernel_size = (5,5), stride = (1,1), padding = (2,2))
    # Max pooling layer
    self.pool = nn.MaxPool2d(kernel_size = (2, 2))
    # First fully connected layer
    self.fc1 = nn.Linear(64 * 4 * 4, 1024)
    # Second fully connected layer
    self.fc2 = nn.Linear(1024, num_classes)
   
  def forward(self,x):
    out = x.reshape(x.shape[0],1,28,28) # Reshape input to 28 x 28 x 1 images
    out = F.relu(self.conv1(out))
    out = F.relu(self.conv2(out))
    out = self.pool(out)
    out = out.reshape(out.shape[0], -1) # Reshape to (number of data, 64*4*4=1028)
    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    return out

model = CNN()
# Loss Function
criterion = nn.CrossEntropyLoss()
# Optimizer (AdamAdamOptimizer, learning rate is 1e-4)
optimizer = optim.Adam(model.parameters(), lr = 10**(-4))
# Define the accuracy
def accuracy(y_true,y_pred):
  label = np.array(y_true)
  prediction = np.array(y_pred.argmax(1))
  output = float(sum(prediction == label)/len(label))
  return output

# Train the model
for epoch in range(2):
  for iteration_num, (data, labels) in enumerate(train_loader):
    data = Variable(data.view(input_shape))
    labels = Variable(labels)
    optimizer.zero_grad()
    scores = model(data)
    loss = criterion(scores, labels)
    loss.backward()
    optimizer.step()
    if iteration_num % 100 == 99:    # Print every 100 iterations
      training_accuracy = accuracy(y_true = labels, y_pred = scores) # Calculate the accuracy of the current iteration
      print(f"Training accuracy (Iteration {1200*epoch + iteration_num + 1}): {training_accuracy:.4f}")
print('Finished Training')


# Calculate the testing accuracy 
with torch.no_grad():
  for (data, labels) in test_loader:
    data = Variable(data.view(input_shape))
    outputs = model(data)
    test_accuracy = accuracy(y_true = labels, y_pred = outputs)
print(f"Test accuracy: {test_accuracy}")

# Visuallising filters
print("Visuallising filters:")
filters = model.conv1.weight.clone().detach().numpy()
filters -= filters.min()
filters /= filters.max()
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(filters[i,0,:,:], cmap='gray')
  plt.xlabel(f"Filter {i+1}")
plt.tight_layout(pad = 0.25)
plt.show()


# Visualising patches with high activation
print("Visualising patches with high activation:")
randomIndex = np.random.permutation(25)
randomIndex = randomIndex[range(5)]

with torch.no_grad():
  out = data.reshape(data.shape[0],1,28,28)
  activation_value = F.relu(model.conv1(out))
activation_value = activation_value.detach().numpy()
activation_value -= activation_value.min()
activation_value /= activation_value.max()

img = out.detach().numpy()
img -= img.min()
img /= img.max()
# Print the patches
indexForSubplot = [1,4,7,9,11]
for i in range(5): # for each randomly picked filter in the first convolutional layer
  # Find indices that sort the activation_value
  coor = np.unravel_index(np.argsort(activation_value[:,randomIndex[i],:,:], axis = None), activation_value[:,randomIndex[i],:,:].shape)
  for j in range(3):
    if (i != 0) and (i != 1) and (j == 2):
      continue
    plt.subplot(4,3,indexForSubplot[i]+j)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # Size of coor : (3,810000)
    sample_index = coor[0][800009-j]
    patch = img[sample_index,0,:,:]
    patch_height = range(2*coor[1][800009-j],2*coor[1][800009-j]+12)
    patch = patch[patch_height,:]
    patch_width = range(2*coor[2][800009-j],2*coor[2][800009-j]+12)
    patch = patch[:,patch_width]
    plt.imshow(patch, cmap='gray')
    plt.xlabel(f"Patch {indexForSubplot[i]+j}")
plt.tight_layout(pad = 0.25)
plt.show()
