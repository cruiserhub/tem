import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter




class MLP(torch.nn.Module):
    def __init__(self, in_dim = 0, hidden_dim = 0, out_dim = 0, bias = [True, True], activation = F.tanh, num_layer = 2):
        # First call super class init function to set up torch.nn.Module style model and inherit it's functionality
        super(MLP, self).__init__()
        # Check if this network consists of module: are input and output dimensions lists? If not, make them (but remember it wasn't)
        # Find number of modules
        # Create weights (input->hidden, hidden->output) for each module
        self.w = torch.nn.ModuleList([])
        self.activation = activation
        self.num_layer = num_layer
        # If number of hidden dimensions is not specified: mean of input and output
        hidden = hidden_dim
            # Each module has two sets of weights: input->hidden and hidden->output
        if self.num_layer>1:
            self.w = torch.nn.ModuleList(
                [torch.nn.Linear(in_dim, hidden, bias=bias[0]), torch.nn.Linear(hidden, out_dim, bias=bias[1])])
        else:
            self.w = torch.nn.ModuleList(
                [torch.nn.Linear(in_dim, out_dim, bias=bias[0])])
        # print (self.w)
        self.relu = nn.ReLU(inplace=False)
        # Copy activation function for hidden layer and output layer
        for from_layer in range(self.num_layer):
            # Set weights to xavier initalisation
            torch.nn.init.xavier_normal_(self.w[from_layer].weight)
            # Set biases to 0
            if bias[from_layer]:
                self.w[from_layer].bias = nn.Parameter(torch.zeros_like(self.w[from_layer].bias, requires_grad=False))

    def set_weights(self, from_layer, value):
        # If single value is provided: copy it for each module
        input_value = value
        # Run through all modules and set weights starting from requested layer to the specified value
        with torch.no_grad():
            # MLP is setup as follows: w[module][layer] is Linear object, w[module][layer].weight is Parameter object for linear weights, w[module][layer].weight.data is tensor of weight values
            self.w[from_layer].weight.fill_(input_value)

    def forward(self, data):
        input_data = data
        # Run input through network for each module
        output = []
        # Pass through first weights from input to hidden layer
        module_output = self.w[0](input_data)
        if self.num_layer == 1:
            return module_output
        # Apply hidden layer activation
        module_output = self.relu(module_output)
        # Pass through second weights from hidden to output layer
        module_output = self.w[1](module_output)
        # Apply output layer activation
        module_output = self.activation(module_output)
        # And return output
        return module_output

if __name__ == '__main__':
    mlp = MLP(5, 5, 5, num_layer=1)
    input = torch.zeros(5)
    output = mlp(input)
    # print (output)