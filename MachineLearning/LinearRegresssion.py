import numpy as np 
import math

#This program is not optimal just understanding of linear regresssion.

Xtrain = [1,2,3,4] #Stores the training inputs
Ytrain = [3,5,7,9] #Stores the training labels

#Hyperparameters 
learningRate = .01
numEpochs =1000

#In this case, our hypothesis is in the form of a model representing univariate 
#linear regression. y = theta0 + x*theta1
def hypothesis(theta0,theta1,x):
	return (theta0 + theta1*x)

#Our loss function is the classic mean squared error form
def costFunction(theta0, theta1):
	loss = 0
	for i, j in zip(Xtrain,Ytrain):
		temp = math.pow((hypothesis(theta0,theta1,i) - j),2)
		loss += temp
	return loss

#Weight updates are done by taking the derivative of the loss function 
#with respect to each of the theta values. 
def weightUpdate(withRespectTo, theta0, theta1):
	if (withRespectTo == "theta0"):
		theta0 = theta0 - learningRate*(derivative(withRespectTo, theta0, theta1))
		return theta0
	else: #has to be wrt to theta1
		theta1 = theta1 - learningRate*(derivative(withRespectTo, theta0, theta1))
		return theta1
	
#Evaluating a numerical deerivative
def derivative(withRespectTo, theta0, theta1):
	h = 1./1000.
	if (withRespectTo == "theta0"):
		rise = costFunction(theta0 + h, theta1) - costFunction(theta0,theta1)
	else: #has to be wrt to theta1
		rise = costFunction(theta0 , theta1 + h) - costFunction(theta0,theta1)
	run = h
	slope = rise/run
	return slope

#Random initialization of the theta values
theta0 = np.random.uniform(0,1)
theta1 = np.random.uniform(0,1)
#Test value
Xtest = 5
for i in range(0,numEpochs):
	theta0 = weightUpdate("theta0", theta0, theta1)
	theta1 = weightUpdate("theta1", theta0, theta1)
print (hypothesis(theta0,theta1,Xtest))
