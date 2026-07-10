import math
import random

def softmax(x):
    m = max(x)
    e = [math.exp(i - m) for i in x]
    s = sum(e)
    return [i / s for i in e]

class SelfAttention:
    def __init__(self, dim):
        self.dim = dim

        self.Wq = [
            [random.uniform(-0.5, 0.5) for _ in range(dim)]
            for _ in range(dim)
        ]

        self.Wk = [
            [random.uniform(-0.5, 0.5) for _ in range(dim)]
            for _ in range(dim)
        ]

        self.Wv = [
            [random.uniform(-0.5, 0.5) for _ in range(dim)]
            for _ in range(dim)
        ]

    def matmul(self, W, x):
        return [
            sum(w * i for w, i in zip(row, x))
            for row in W
        ]

    def forward(self, x_seq):
        Q = [self.matmul(self.Wq, x) for x in x_seq]
        K = [self.matmul(self.Wk, x) for x in x_seq]
        V = [self.matmul(self.Wv, x) for x in x_seq]

        outputs = []

        for i in range(len(x_seq)):
            scores = []

            for j in range(len(x_seq)):
                score = sum(
                    Q[i][k] * K[j][k]
                    for k in range(self.dim)
                )
                score /= math.sqrt(self.dim)
                scores.append(score)
            weights = softmax(scores)

            out = [0.0] * self.dim

            for j in range(len(x_seq)):
                for d in range(self.dim):
                    out[d] += weights[j] * V[j][d]

            outputs.append(out)

        return outputs