# monitor/visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_coherence(history, out_dir: Path):
    xs = np.arange(len(history))
    if len(xs) == 0:
        print("⚠️ No history to plot — training may have failed early.")
        return
    
    trust_e = np.array([h.trust_emotion for h in history])
    trust_r = np.array([h.trust_reason for h in history])
    harmony = np.array([h.harmony for h in history])
    reward = np.array([h.reward for h in history])

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(xs, reward, label="Reward", color="tab:blue")
    ax1.plot(xs, harmony, label="Harmony", color="tab:orange")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(xs, trust_e, "--", label="Trust (Emotion)", color="tab:red")
    ax2.plot(xs, trust_r, "--", label="Trust (Reason)", color="tab:green")
    ax2.set_ylabel("Trust")
    ax2.legend(loc="upper right")

    plt.title("SynMind: Coherence and Trust Evolution")
    plt.tight_layout()
    plt.savefig(out_dir / "coherence_trust_evolution.png", dpi=150)
    plt.close()
