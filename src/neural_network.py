from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self,neurons=[1,1,1]):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(neurons[0], neurons[1]), #1st layer: pass the data of x_0 points to x_1 neurons
            nn.ReLU(),
            nn.Linear(neurons[1], neurons[1]), #hidden layers: x_i neurons to x_i=1 neurons
            nn.ReLU(),
            nn.Linear(neurons[1], neurons[2]) #last layer: x_-2 to x_-1 neurons
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits