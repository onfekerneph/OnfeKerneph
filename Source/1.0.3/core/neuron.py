import random

class Neuron:
    def __init__(self, n_inputs):
        self.w = [random.uniform(-0.5,0.5) for _ in range(n_inputs)]
        self.b = random.uniform(-0.5,0.5)

    def forward(self, x):
        self.x = x
        self.out = sum(i*j for i,j in zip(x,self.w)) + self.b
        return self.out