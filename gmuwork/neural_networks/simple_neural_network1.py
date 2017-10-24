import random
import math
import numpy as np
#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively
#
# Comment references:
#
# [1] Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#   https://class.coursera.org/neuralnets-2012-001/lecture/39
# [3] The Back Propagation Algorithm
#   https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

class simpleNeuralNetwork:

    def __init__(self,input_dim,output_dim,epsilon=.01,reg_lambda=.01):
        self.input_dim =input_dim
        self.output_dim = output_dim
        self.epsilon =epsilon
        self.reg_lambda = reg_lambda
    def calculate_loss(self):
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        # Forward propagation to calculate our predictions
        z1 = self.X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(self.num_examples), self.y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./self.num_examples * data_loss
    def train(self,X,y,nn_hdim, num_passes=20000, print_loss=False):
        self.X =np.array(X)
        self.y =np.array(y)
        self.num_examples = len(X)
        np.random.seed(0)
        W1 = np.random.randn(self.input_dim, nn_hdim) / np.sqrt(self.input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, self.output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, self.output_dim))
        # This is what we return at the end
        model = {}
        
        # Gradient descent. For each batch...
        for i in range(0, num_passes):
    
            # Forward propagation
            z1 = np.dot(X,W1)  + b1
            a1 = np.tanh(z1)
            z2 = np.dot(a1,W2)  + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
            # Backpropagation
            delta3 = probs
            delta3[range(self.num_examples), y] -= 1
            dW2 = np.dot(a1.T,delta3) 
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = np.dot(delta3,W2.T)  * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)
    
            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * W2
            dW1 += self.reg_lambda * W1
    
            # Gradient descent parameter update
            W1 += -self.epsilon * dW1
            b1 += -self.epsilon * db1
            W2 += -self.epsilon * dW2
            b2 += -self.epsilon * db2
            self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
            # Assign new parameters to the model            
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" %(i, self.calculate_loss()))
        def predict(self,x):
            W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
            # Forward propagation
            z1 = np.dot(x,W1)  + b1
            a1 = np.tanh(z1)
            z2 = np.dot(a1,W2)  + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return np.argmax(probs, axis=1)
if __name__ =="__main__":
    nn = simpleNeuralNetwork(2,2)
    nn.train([1,10],[0,1],3)