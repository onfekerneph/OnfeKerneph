from core.embedding import Embedding
from core.attention import SelfAttention
from utils.math_utils import softmax, dot, cross_entropy

import math
import random
import pickle


class Network:
    def __init__(self,vocab_size=4,dim=16,lr=0.005,ff_hidden=128):
        self.vocab = vocab_size
        self.dim = dim
        self.lr = lr
        self.ff_hidden = ff_hidden
        self.embed = Embedding(vocab_size, dim)

        # Multi-head Attention
        self.attn1 = SelfAttention(dim)
        self.attn2 = SelfAttention(dim)

        # Feed Forward
        self.ff1 = [
            [
                random.uniform(-0.02, 0.02)
                for _ in range(ff_hidden)
            ]
            for _ in range(dim)
        ]

        self.ff2 = [
            [
                random.uniform(-0.02, 0.02)
                for _ in range(dim)
            ]
            for _ in range(ff_hidden)
        ]

        # Output Layer
        self.output = [
            [
                random.uniform(-0.01, 0.01)
                for _ in range(dim)
            ]
            for _ in range(vocab_size)
        ]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def positional_encoding(self, emb, pos):
        result = emb[:]
        for i in range(self.dim):
            if i % 2 == 0:
                result[i] += math.sin(pos /(100000 ** (i / self.dim)))
            else:
                result[i] += math.cos(pos /(100000 ** ((i - 1) / self.dim)))
        return result

    def feed_forward(self, vec):
        hidden = []
        for h in range(self.ff_hidden):
            s = 0.0
            for d in range(self.dim):
                s += vec[d] * self.ff1[d][h]
            hidden.append(max(0, s))
        output = []
        for d in range(self.dim):
            s = 0.0
            for h in range(self.ff_hidden):
                s += hidden[h] * self.ff2[h][d]
            output.append(s)
        return output, hidden

    def forward(self, seq):
        x = []
        # Embedding + Positional Encoding
        for pos, token in enumerate(seq):
            emb = self.embed.forward(token)
            emb = self.positional_encoding(emb,pos)
            x.append(emb)
        # Attention Head 1
        a1 = self.attn1.forward(x)
        # Attention Head 2
        a2 = self.attn2.forward(x)
        merged = []
        for i in range(len(a1)):
            vec = []
            for v1, v2 in zip(a1[i], a2[i]):
                vec.append((v1 + v2) / 2)
            merged.append(vec)
        # Son token temsili
        last = merged[-1]
        # Feed Forward
        ff_input = last

        last, hidden = self.feed_forward(last)

        logits = [
            dot(last, w)
            for w in self.output
        ]

        probs = softmax(logits)
        return probs, last, seq, hidden, ff_input

    def train_step(self, seq, target):
        probs, last, seq_ids, hidden, ff_input = self.forward(seq)
        loss = cross_entropy(probs,target)
        grad = probs[:]
        grad[target] -= 1
        grad_last = [0.0] * self.dim
        
        for k in range(self.vocab):
            for d in range(self.dim):
                grad_last[d] += (grad[k] * self.output[k][d])
        for i in range(len(grad_last)):
            if grad_last[i] > 1.0:
                grad_last[i] = 1.0
            elif grad_last[i] < -1.0:
                grad_last[i] = -1.0
        # Output gradient
        emb_grad = [0.0] * self.dim
        for k in range(self.vocab):
            for d in range(self.dim):
                emb_grad[d] += (grad[k] * self.output[k][d])

        # Output update
        for k in range(self.vocab):
            for d in range(self.dim):
                self.output[k][d] -= (self.lr * grad[k] * last[d])
        grad_hidden = [0.0] * self.ff_hidden
        
        # DÜZELTME — sırayı tersine çevir
        # Önce grad_hidden hesapla
        
        for h in range(self.ff_hidden):
            for d in range(self.dim):
                grad_hidden[h] += (grad_last[d] * self.ff2[h][d])
            if hidden[h] <= 0:
                grad_hidden[h] = 0

        # Sonra ff2'yi güncelle
        for h in range(self.ff_hidden):
            for d in range(self.dim):
                self.ff2[h][d] -= (self.lr * hidden[h] * grad_last[d])
        for i in range(len(grad_hidden)):
            if grad_hidden[i] > 1.0:
                grad_hidden[i] = 1.0
            elif grad_hidden[i] < -1.0:
                grad_hidden[i] = -1.0

        for d in range(self.dim):
            for h in range(self.ff_hidden):
                self.ff1[d][h] -= (self.lr * ff_input[d] * grad_hidden[h])
        
        # Embedding update
        for idx in seq_ids:
            self.embed.update(idx,emb_grad,self.lr)
        return loss

    def train(self,data,epochs=2000,step=100):
        logs = []
        summ = []
        ls = []
        m = 0.0
        tot = 0
        width = len(str(epochs))
        for epoch in range(epochs):
            total = 0.0
            for seq, target in data:
                total += self.train_step(seq,target)
            tot = total
            summ.append(total)
            progress = int(epoch / epochs * 50)
            bar = "#" * progress
            print(
                f"\r[{bar:<50}] "
                f"%{math.floor(epoch/epochs*100)} "
                f"epoch:{epoch} "
                f"loss:{round(total,4)}",
                f"plusmod:{round(m,5)}",
                end="",
                flush=True,
            )

            if epoch % step == 0:
                logs.append(
                    f"epoch:{epoch:<{width}} "
                    f"loss:{round(total,5)} "
                    f"plusmod:{round(m,5)}"
                )

            ls.append(total)
            if min(ls) < total:
                if (total - min(ls)) > m:
                    m = (total - min(ls))
            if epoch % step == 0:
                logs.append(
                    f"epoch:{epoch:<{width}} "
                    f"loss:{round(total,4)} "
                    f"plusmod:{round(m,4)}"
                )
                

        print("\n")
        print("\n".join(logs))
        return (sum(summ)/len(summ))

    def predict(self, seq):
        probs, _, _,_,_ = self.forward(seq)
        return max(
            range(len(probs)),
            key=lambda i: probs[i]
        )
    