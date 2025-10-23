import numpy as np
from collections import deque

class MemorySystem:
    """
    Simple STM/LTM with exponential decay and consolidation.
    - STM keeps last K stimuli vectors.
    - LTM keeps running prototypes (mean + count) for two buckets:
        'emotion' and 'reason'.
    """
    def __init__(self, dim: int, stm_len: int = 16, decay: float = 0.96):
        self.dim = dim
        self.stm = deque(maxlen=stm_len)
        self.decay = decay
        # LTM prototypes
        self.protos = {
            "emotion": {"mu": np.zeros(dim, dtype=np.float32), "n": 1e-6},
            "reason":  {"mu": np.zeros(dim, dtype=np.float32), "n": 1e-6},
        }

    def push_stm(self, x: np.ndarray):
        self.stm.append(x.astype(np.float32))

    def stm_vector(self) -> np.ndarray:
        if not self.stm:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(np.stack(self.stm, axis=0), axis=0)

    def update_ltm(self, x: np.ndarray, bank: str):
        bank = "emotion" if bank not in self.protos else bank
        proto = self.protos[bank]
        proto["mu"] = self.decay * proto["mu"] + (1 - self.decay) * x
        proto["n"]  = proto["n"] + 1

    def get_proto(self, bank: str) -> np.ndarray:
        return self.protos[bank]["mu"].copy()

    def snapshot(self):
        return {
            "stm_len": len(self.stm),
            "proto_emotion_norm": float(np.linalg.norm(self.protos["emotion"]["mu"])),
            "proto_reason_norm":  float(np.linalg.norm(self.protos["reason"]["mu"])),
        }
