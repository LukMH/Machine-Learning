# COMP5541 Assignment 1
# Q3 Naive Bayes Classifier
import numpy as np
from scipy import stats
from sklearn.datasets import load_iris
from math import sqrt
iris = load_iris()
X, y = iris['data'], iris['target']

# Data preparation
N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]

# NBC class
class NBC:
  def __init__(self, feature_types, num_classes):
    self.feature_types = feature_types
    self.num_classes = num_classes
    self.prior_prob = None
    self.para = None

  def fit(self, X_train, y_train):
    # Compute the prior distribution
    n_train, d_train = X_train.shape
    prior_prob = np.zeros((self.num_classes), dtype="double")
    for i in range(self.num_classes):
      prior_prob[i] = sum(y_train==i)/n_train
      print("Prior probability (Class: "+str(i)+") : "+str(prior_prob[i]))
    self.prior_prob = prior_prob

    # Compute parameters of the conditional distribution
    para = np.zeros((d_train, self.num_classes * 2), dtype="double")
    for i in range(self.num_classes):
      para[:,2*i] = np.mean(X_train[y_train==i], axis=0) # Compute mean
      para[:,2*i+1] = np.var(X_train[y_train==i], axis=0) # Compute variance
      # Ensure variance is nonzero
      para[:,2*i+1] = para[:,2*i+1] + (10**(-6))*(para[:,2*i+1] == 0)
      for j in range(d_train):
        print("Conditional distribution of feature "+str(j+1)+" given class = "+str(i)+" : N("+str(para[j,2*i])+", "+str(para[j,2*i+1])+")")
    self.para = para

  def predict(self, X_test):
    N_test, D_test = X_test.shape
    post_prob = np.zeros((N_test, self.num_classes),dtype="double")
    for i in range(self.num_classes):
      product_feat_prob = np.ones((N_test,1),dtype="double")
      # Get the product of feature probability
      for j in range(D_test):
        product_feat_prob = product_feat_prob * stats.norm.pdf(X_test[:,j],self.para[j,2*i],sqrt(self.para[j,2*i+1])).reshape((N_test,1))
      # Get the posterior class probability
      post_prob[:,i] = self.prior_prob[i]*product_feat_prob.reshape((1,N_test))
    yhat = np.argmax(post_prob, axis=1) # Return the class that has the largest probability
    return yhat



nbc = NBC(feature_types = ['r','r','r','r'], num_classes = 3)
nbc.fit(Xtrain, ytrain)

yhat_train = nbc.predict(Xtrain)
print("Prediction of Xtrain points:")
print(yhat_train)
print("Actual class of Xtrain points:")
print(ytrain)
train_accuracy = np.mean(yhat_train == ytrain) 
print("Training accuracy: "+str(train_accuracy))

yhat = nbc.predict(Xtest)
print("Prediction of Xtest points:")
print(yhat)
print("Actual class of Xtest points:")
print(ytest)
test_accuracy = np.mean(yhat == ytest) 
print("Testing accuracy: "+str(test_accuracy))
