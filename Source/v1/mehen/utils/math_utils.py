#math_utils.py
import math

def softmax(x):
    m = max(x)
    e = [math.exp(i - m) for i in x]
    s = sum(e)
    return [i/s for i in e]

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def cross_entropy(probs, target):
    return -math.log(max(probs[target], 1e-9))