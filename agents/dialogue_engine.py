import random
import numpy as np
from dataclasses import dataclass

@dataclass
class DialogueTurn:
    emotion_msg: str
    reason_msg: str
    harmony_delta: float

class DialogueEngine:
    def __init__(self):
        self.states = ["joyful", "tense", "neutral", "curious", "conflicted", "focused"]
        self.templates_emotion = [
            "I feel {state} about this.",
            "Something feels {state}.",
            "Emotionally, it's quite {state}.",
            "My intuition says this is {state}.",
        ]
        self.templates_reason = [
            "Logically, this seems {state}.",
            "From a reasoning standpoint, it's {state}.",
            "Let's analyze why it feels {state}.",
            "It might indeed be {state}.",
        ]
        self.tone_noise = 0.05

    def exchange(self, affect: float, decision: float) -> DialogueTurn:
        diff = abs(affect - decision)
        state_idx = int(diff * (len(self.states) - 1))
        state_idx = max(0, min(state_idx, len(self.states) - 1))  # clamp index

        state = self.states[state_idx]
        emo_msg = random.choice(self.templates_emotion).format(state=state)
        reason_msg = random.choice(self.templates_reason).format(state=state)

        base_harmony = 1.0 - abs(affect - decision)
        harmony_delta = base_harmony - 0.5 + random.uniform(-self.tone_noise, self.tone_noise)

        return DialogueTurn(
            emotion_msg=emo_msg,
            reason_msg=reason_msg,
            harmony_delta=float(np.clip(harmony_delta, -1, 1))
        )
