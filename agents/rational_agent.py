# agents/rational_agent.py
import numpy as np
from dataclasses import dataclass

@dataclass
class RationalOutput:
    decision: float
    certainty: float
    qvalue: float

class RationalAgent:
    """
    The 'Reason' subsystem — predicts rational outcomes based on emotional bias and environment.
    Includes gradient clipping and homeostatic regulation to prevent overflow.
    """
    def __init__(self, dim, lr=0.03, reg=0.001):
        self.theta = np.random.uniform(-0.1, 0.1, size=(dim,))
        self.bias = 0.0
        self.lr = lr
        self.reg = reg

    def forward(self, x, emotional_bias=0.0, meta=None):
        x_eff = np.clip(x + emotional_bias * 0.1, -2, 2)
        yhat = float(np.tanh(np.clip(self.theta @ x_eff, -10, 10)))
        certainty = 1.0 - np.mean(np.abs(x_eff)) * 0.2
        qvalue = yhat * certainty

        if np.isnan(yhat) or np.isnan(certainty) or np.isnan(qvalue):
            yhat, certainty, qvalue = 0.0, 0.5, 0.0

        return RationalOutput(
            decision=float(np.clip(yhat, -1, 1)),
            certainty=float(np.clip(certainty, 0, 1)),
            qvalue=float(np.clip(qvalue, -1, 1))
        )

    def learn(self, x, emotional_bias, target_value):
        x_eff = np.clip(x + emotional_bias * 0.1, -2, 2)
        pred = np.tanh(np.clip(self.theta @ x_eff, -10, 10))
        err = np.clip(target_value - pred, -2, 2)
        grad = np.clip(x_eff, -2, 2)

        self.theta += np.clip(self.lr * (err * grad - self.reg * self.theta), -1, 1)
        self.bias = np.clip(self.bias + 0.01 * err, -1, 1)
        self.theta = np.clip(self.theta, -5, 5)
        self.lr = max(1e-4, min(self.lr, 0.1))
