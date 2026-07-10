import random

class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = []

        for _ in range(n_out):
            self.neurons.append({
                "w": [random.uniform(-0.5, 0.5) for _ in range(n_in)],
                "b": random.uniform(-0.5, 0.5),
                "x": None
            })

    def forward(self, seq):
        outputs = []

        # 🔥 HER TIMESTEP AYRI
        for x in seq:
            out_vec = []

            for n in self.neurons:
                n["x"] = x
                out = sum(i * j for i, j in zip(x, n["w"])) + n["b"]
                out_vec.append(out)

            outputs.append(out_vec)

        return outputs

    def update(self, grad_seq, lr):

        # güvenlik: min length
        T = min(len(grad_seq), len(self.neurons))

        for t in range(T):
            grad_t = grad_seq[t]

            # neuron sayısı ile eşleştir
            N = min(len(grad_t), len(self.neurons))

            for i in range(N):
                n = self.neurons[i]

                for j in range(len(n["w"])):
                    n["w"][j] -= lr * grad_t[i] * n["x"][j]

                n["b"] -= lr * grad_t[i]