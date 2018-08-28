import datetime
import os
import sys
import argparse
import torch
from torch import Tensor, LongTensor
from torch.autograd import Variable

import IPython
import pickle


class Solver(object):

    def __init__(self, net, data):


        self.net = net
        self.data = data

        # Number of iterations to train for
        self.max_iter = 5000
        # Every this many iterations, record accuracy
        self.summary_iter = 200




        '''
        Use Stochastic Gradient Descent with momentum
        '''

        self.optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    def optimize(self):
        # Record both of these accuracies every self.summary_iter iterations
        self.train_accuracy = []
        self.test_accuracy = []

        '''
        Trains the network
        '''
        for i in range(self.max_iter):
            X_batch, y_batch = self.data.get_train_batch()
            X_batch, y_batch = Variable(Tensor(X_batch)), Variable(LongTensor(y_batch))
            self.optimizer.zero_grad() #clear gradients
            predictions = self.net.net(X_batch)
            loss = self.net.loss_fn(predictions, y_batch)
            loss.backward() #back-prop
            self.optimizer.step() #update parameters
            if i % self.summary_iter == 0:
                print(i)
                X_val, y_val = self.data.get_validation_batch()
                X_val, y_val = Variable(Tensor(X_val)), Variable(LongTensor(y_val))
                train_accuracy = self.net.accuracy(X_batch, y_batch)
                val_accuracy = self.net.accuracy(X_val, y_val)
                self.train_accuracy.append(train_accuracy)
                self.test_accuracy.append(val_accuracy)
