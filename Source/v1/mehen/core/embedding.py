import random

class Embedding:
    def __init__(self, vocab_size, dim):
        self.vocab_size = vocab_size
        self.dim = dim
        self.table = [[random.uniform(-0.5, 0.5) for _ in range(dim)]for _ in range(vocab_size)]

    def forward(self, idx):
        return self.table[idx].copy()

    def update(self, idx, grad, lr):
        for i in range(self.dim):
            self.table[idx][i] -= lr * grad[i]