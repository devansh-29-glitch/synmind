import numpy as np
from dataclasses import dataclass

@dataclass
class EmotionalOutput:
    affect: float
    bias: float

class EmotionalAgent:
    """
    The 'Emotion' subsystem — generates affective bias signals.
    Uses a Hebbian-like update rule for associative learning.
    """
    def __init__(self, dim, lr=0.05, reg=0.001):
        self.w = np.random.uniform(-0.1, 0.1, size=(dim,))
        self.bias = 0.0
        self.lr = lr
        self.reg = reg

    def forward(self, x, meta):
        a = float(np.tanh(np.clip(self.w @ x + self.bias, -10, 10)))
        return EmotionalOutput(affect=a, bias=self.bias)

    def learn(self, x, target_affect):
        pred = self.forward(x, meta={}).affect
        err = np.clip(target_affect - pred, -1, 1)
        grad = np.clip(x, -2, 2)

        self.w += np.clip(self.lr * (err * grad - self.reg * self.w), -1, 1)
        self.bias = np.clip(self.bias + 0.01 * err, -1, 1)

        self.w = np.clip(self.w, -5, 5)
        if np.isnan(self.w).any():
            self.w = np.random.uniform(-0.1, 0.1, size=self.w.shape)
