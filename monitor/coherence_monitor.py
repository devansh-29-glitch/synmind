import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

def plot_training_curves(history: List, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    xs = np.arange(len(history))
    reward  = np.array([h.reward   for h in history])
    harmony = np.array([h.harmony  for h in history])
    affect  = np.array([h.affect   for h in history])
    decide  = np.array([h.decision for h in history])
    cert    = np.array([h.certainty for h in history])

    plt.figure(figsize=(10,4))
    plt.plot(xs, reward,  label="reward")
    plt.plot(xs, harmony, label="harmony")
    plt.plot(xs, cert,    label="certainty")
    plt.legend(); plt.title("SynMind — Reward/Harmony/Certainty"); plt.xlabel("step")
    plt.tight_layout(); plt.savefig(out_dir / "curves_reward_harmony_certainty.png", dpi=150); plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(xs, affect, label="affect (S1)")
    plt.plot(xs, decide, label="decision (S2)")
    plt.legend(); plt.title("SynMind — Affect vs Decision"); plt.xlabel("step")
    plt.tight_layout(); plt.savefig(out_dir / "curves_affect_decision.png", dpi=150); plt.close()
