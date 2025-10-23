# trainer/symbiosis_trainer.py
import numpy as np
from dataclasses import dataclass
# ✅ Do NOT import this file again — only import from agents and reward_manager
from agents.dialogue_engine import DialogueEngine
from trainer.reward_manager import RewardManager  # safe one-way import only

@dataclass
class StepLog:
    affect: float
    decision: float
    certainty: float
    qvalue: float
    harmony: float
    reward: float
    trust_emotion: float
    trust_reason: float
    emotion_msg: str
    reason_msg: str

class SymbiosisTrainer:
    """
    Orchestrates the co-training of two neural agents — one emotional, one rational.
    Handles their dialogue exchange, reward feedback, and trust adaptation.
    """
    def __init__(self):
        self.dialogue = DialogueEngine()
        self.rewards = RewardManager()
        self.history = []

    @staticmethod
    def harmony(affect: float, decision: float) -> float:
        if np.isnan(affect) or np.isnan(decision):
            return 0.0
        return 1.0 - abs(affect - decision) / 2.0

    def step(self, emo, rat, stim):
        # forward passes
        emo_out = emo.forward(stim.vec, stim.meta)
        rat_out = rat.forward(stim.vec, emo_out.bias, stim.meta)

        # dialogue interpretation
        turn = self.dialogue.exchange(emo_out.affect, rat_out.decision)

        # harmony & reward computation
        h = self.harmony(emo_out.affect, rat_out.decision)
        task_signal = float(np.clip(stim.meta["importance"] - stim.meta["uncertainty"], -1, 1))
        reward = self.rewards.compute(h, task_signal, turn.harmony_delta)

        # learning rate modulation (trust adaptation)
        emo.lr *= self.rewards.trust_emotion
        rat.lr *= self.rewards.trust_reason

        # learning updates
        emo.learn(stim.vec, target_affect=reward)
        rat.learn(stim.vec, emo_out.bias, target_value=reward)

        # logging
        log = StepLog(
            affect=emo_out.affect,
            decision=rat_out.decision,
            certainty=rat_out.certainty,
            qvalue=rat_out.qvalue,
            harmony=h,
            reward=reward,
            trust_emotion=self.rewards.trust_emotion,
            trust_reason=self.rewards.trust_reason,
            emotion_msg=turn.emotion_msg,
            reason_msg=turn.reason_msg,
        )
        self.history.append(log)
        return log