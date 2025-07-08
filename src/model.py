
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model,self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)       # fully connected layer1  feed the input into 128 neurons (first hidden layer)
        self.fc2 = nn.Linear(128,64)                # second hidden layer
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()                       # break linearity | rectified linear unit - activation function k = max(0,k)
        self.dropout = nn.Dropout(0.5)              # dropout function 0.5 probability of something not be included

    def forward(self, x):
        x = self.relu(self.fc1(x))                  # weights and biases calculations, then apply ReLU
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)                             # dont apply ReLU, because these are the logits for softmax
                                                    # if a logit is negative it's important information - model thinks the input is not part of the classification
        return x