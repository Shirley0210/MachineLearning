import numpy as np 
import math

#basic implimentation of logistic regresssion and it is not optimal code.
# But it gives sense of 	
#  backpropagation, 
#  computing the loss function,
#  updating the weights. 
#The derivatives are taken numerically, instead of analytically. 

Xtrain = [5,6,10,7,4]
Ytrain = [1,1,0,0,1] #Stores the training labels

#Hyperparameters 
numTrainingExamples = 5
learningRate = .1
numEpochs =1000

#In this case, our hypothesis is in the form of a model representing logistic
#regression
def hypothesis(theta0,theta1,x):
	return (1/(1+np.exp(-(theta0 + theta1*x))))

#Our loss function is different depending on whether the y value is 0 or 1
def costFunction(theta0, theta1):
	loss = 0
	for i, j in zip(Xtrain,Ytrain):
		temp = (-j*math.log(hypothesis(theta0,theta1,i))) - (1-j)*math.log(1 - hypothesis(theta0,theta1,i))
		loss += temp
	return loss/numTrainingExamples

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
