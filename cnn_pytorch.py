import torch
import torch.nn as nn #neural network functionality
import torch.nn.functional as F #things for neural nets that are just functions, not objects
import numpy as np

import IPython

class CNN_PyTorch(nn.Module):
    def __init__(self, num_outputs):
        super(CNN_PyTorch, self).__init__()
        '''
        Initialize layers
        '''
        self.conv = nn.Conv2d(3, 5, kernel_size=15, padding=7)#input channels, output channels, filter size
        self.pool = nn.MaxPool2d(3)
        self.lin1 = nn.Linear(4500, 512)
        self.lin2 = nn.Linear(512, num_outputs)

    def forward(self, x):
        '''
        Compute forward pass
        '''
        self.conv_out = self.conv(x)
        x = F.relu(self.conv_out)
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x)) #reshape
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x

    def response_map(self, x):
        '''
        Return the response Variable from the conv
        '''
        return self.conv_out

    def num_flat_features(self, x): #this computes the number of elements in a multidimensional variable (ignoring batch)
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
class CNN(object):

    def __init__(self,classes,image_size):
        '''
        Initializes the size of the network
        '''

        self.classes = classes
        self.num_class = len(self.classes)
        self.image_size = image_size

        self.output_size = self.num_class
        self.batch_size = 40

        self.net = self.build_network(num_outputs=self.output_size)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def build_network(self, num_outputs):
        return CNN_PyTorch(num_outputs)

    def get_acc(self,y_,y_out):

        '''
        Compute network accuracy
        '''
        _, predicted = torch.max(y_out, 1)
        total = len(y_)
        correct = (predicted == y_).sum().data.numpy()
        accuracy = correct / total
        return accuracy

    def accuracy(self, images, y_):
        return self.get_acc(y_, self.net(images))
    def parameters(self):
        return self.net.parameters()
    def response_map(self, x):
        return self.net.response_map(x)
