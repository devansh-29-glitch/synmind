import numpy as np
from pathlib import Path
from tqdm import trange

from core.environment import Environment
from agents.emotional_agent import EmotionalAgent
from agents.rational_agent import RationalAgent
from trainer.symbiosis_trainer import SymbiosisTrainer
from monitor.coherence_monitor import plot_training_curves
from monitor.visualizer import visualize_coherence  # <-- Added this import

ASSETS = Path("assets")

def main():
    env = Environment(seed=7)
    emo = EmotionalAgent(dim=env.dim, lr=0.05)
    rat = RationalAgent(dim=env.dim, lr=0.03)
    coach = SymbiosisTrainer()

    print("\n🧠 Reinforcement learning begins...\n")
    for step in trange(600, desc="🤝 Adaptive Co-Training"):  # Updated loop to run for 600 steps
        stim = env.sample()
        log = coach.step(emo, rat, stim)
        if step % 120 == 0:  # Changed print frequency to every 120 steps
            print(f"\nEmotion: {log.emotion_msg}")
            print(f"Reason:  {log.reason_msg}")
            print(f"Harmony: {log.harmony:.2f}, Reward: {log.reward:.2f}")
            print(f"Trust — Emotion: {log.trust_emotion:.2f}, Reason: {log.trust_reason:.2f}")

    plot_training_curves(coach.history, ASSETS)
    visualize_coherence(coach.history, ASSETS)  # <-- Added coherence visualization
    print("\n✅ Reinforcement phase complete. Visuals saved in assets/")

if __name__ == "__main__":
    main()
