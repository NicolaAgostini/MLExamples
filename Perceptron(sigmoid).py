
import numpy as np;

import pandas as pd;

! git clone --recursive https://github.com/architsingh15/Perceptron-Algorithm-from-scratch-Sonar-Dataset.git;

dataset = pd.read_csv('Perceptron-Algorithm-from-scratch-Sonar-Dataset/sonar.all-data.csv').as_matrix(); #read csv dataset



X = np.genfromtxt('Perceptron-Algorithm-from-scratch-Sonar-Dataset/sonar.all-data.csv', delimiter=',', dtype=float); #import all the float data

XT=X[:,:-1] #keep only float data

Y = np.genfromtxt('Perceptron-Algorithm-from-scratch-Sonar-Dataset/sonar.all-data.csv', delimiter=',', dtype=str)#import string data which are the target values

YR = Y[:,-1] #keep only the target value

YT=np.zeros(len(YR)); #assign 1 to "Rock" and -1 to "Metal"
for i in range(len(YR)):
  if YR[i]=="R":
    YT[i]=1;
  else:
    YT[i]=-1;

def net_input(xi,W): #net input between a row and weights as in the perceptron formula with linear activation (hard threshold)
  return np.dot(xi,W[1:])+W[0];

def perceptron_predict(xi, W): #predict target of datas
   return np.where(net_input(xi,W)>=0,1,-1)

def train(X,W,Y,eta,n_ite): #train model with linear activation
  for i in range(n_ite):
    for i ,xi in enumerate(X):
      new_w = eta * (Y[i] - perceptron_predict(xi,W));
      W[1:] += new_w*xi;
      W[0] += new_w;

W = np.random.rand(61); #generate random weights between 0 and 1

train(XT,W,YT,0.001,150); #train the model

errors=0;
for i in range(151,206):    #calculate errors %
  if perceptron_predict(XT[i],W) != YT[i]:
    errors+=1;
print("Error rate: ",errors/(206-151)*100,"%");

def trainS(X,W,Y,eta,n_ite): #perceptron with sigmoid activation
  for i in range(n_ite):
    for i ,xi in enumerate(X):
      new_w = eta * (Y[i] - perceptron_predictS(xi,W));
      W[1:] += new_w*xi;
      W[0] += new_w;

def perceptron_predictS(xi, W): #Sigmoid activation
   return np.where(net_inputS(xi,W)>=0.5,1,-1)

def net_inputS(xi,W): #sigmoid
  return 1.0 / (1.0 + np.exp(-(np.dot(xi,W[1:])+W[0])));

WS = np.random.rand(61);

trainS(XT,WS,YT,0.1,150); #with very little eta parameter error rate grows

errorsS=0;
for i in range(151,206): #errors % in sigmoid case
  if perceptron_predictS(XT[i],WS) != YT[i]:
    errorsS+=1;
print("Error rate: ",errorsS/(206-151)*100,"%");

