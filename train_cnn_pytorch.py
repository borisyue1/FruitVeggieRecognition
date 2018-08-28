from data_manager_pytorch import data_manager
from cnn_pytorch import CNN
from trainer_pytorch import Solver
import random


import matplotlib.pyplot as plt

CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape','orange','strawberry','pineapple',
    'radish','carrot','potato','tomato','bellpepper','broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']


image_size = 90

random.seed(0)

classes = CLASS_LABELS
dm = data_manager(classes, image_size)

cnn = CNN(classes,image_size)

solver = Solver(cnn,dm)

solver.optimize()

plt.plot(solver.test_accuracy,label = 'Validation')
plt.plot(solver.train_accuracy, label = 'Training')
plt.legend()
plt.xlabel('Iterations (in 200s)')
plt.ylabel('Accuracy')
plt.show()


