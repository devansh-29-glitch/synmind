
import numpy as np
from collections import deque

class RewardManager:
    """
    Dynamic reinforcement engine for SynMind.
    - Tracks moving averages of harmony & reward.
    - Adjusts emotional and rational learning trust weights.
    - Computes delayed Q-like rewards.
    """
    def __init__(self, gamma=0.92, w_harmony=0.5, w_task=0.3, w_dialogue=0.2):
        self.gamma = gamma
        self.w_harmony = w_harmony
        self.w_task = w_task
        self.w_dialogue = w_dialogue
        self.prev_reward = 0.0
        self.recent_rewards = deque(maxlen=30)
        self.trust_emotion = 1.0
        self.trust_reason = 1.0

    def compute(self, base_harmony, task_signal, dialogue_delta):
        reward = (
            self.w_harmony * base_harmony
            + self.w_task * task_signal
            + self.w_dialogue * (0.5 + 0.5 * dialogue_delta)
        )

        # Add a discount from previous reward (reinforcement-like)
        reward = reward + self.gamma * self.prev_reward
        self.prev_reward = reward
        self.recent_rewards.append(reward)

        # Adjust trust levels dynamically
        avg_r = np.mean(self.recent_rewards)
        if avg_r > 0.6:
            self.trust_emotion = min(1.2, self.trust_emotion + 0.01)
            self.trust_reason  = min(1.2, self.trust_reason + 0.01)
        elif avg_r < 0.3:
            self.trust_emotion = max(0.7, self.trust_emotion - 0.02)
            self.trust_reason  = max(0.7, self.trust_reason - 0.02)

        return float(np.clip(reward, -1, 1))
