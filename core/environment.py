from dataclasses import dataclass
import numpy as np

@dataclass
class Stimulus:
    """A multi-dimensional synthetic 'situation' the agents respond to."""
    vec: np.ndarray                 # shape (D,)
    meta: dict                      # any metadata (uncertainty, novelty, etc.)

class Environment:
    """
    Emits multi-feature stimuli:
      [valence, arousal, logic_load, ambiguity, novelty, fatigue]
    Ranges are normalized to [-1, 1] (except novelty ∈ [0,1]).
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.dim = 6

    def sample(self) -> Stimulus:
        valence     = self.rng.uniform(-1.0, 1.0)       # mood sign
        arousal     = self.rng.uniform(-1.0, 1.0)       # energy / stress
        logic_load  = self.rng.uniform(-1.0, 1.0)       # reasoning demand
        ambiguity   = self.rng.uniform(-1.0, 1.0)       # clarity vs ambiguity
        novelty     = self.rng.uniform(0.0,  1.0)       # how new it is
        fatigue     = self.rng.uniform(-1.0, 1.0)       # tiredness

        x = np.array([valence, arousal, logic_load, ambiguity, novelty, fatigue], dtype=np.float32)

        meta = {
            "uncertainty": float(abs(ambiguity) * (0.3 + 0.7 * (1.0 - novelty))),
            "importance":  float(0.4 + 0.6 * (abs(valence) + abs(logic_load)) / 2.0),
        }
        return Stimulus(vec=x, meta=meta)
