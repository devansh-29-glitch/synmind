# dashboard/make_clip.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

LOGS = Path("logs") / "synmind_stream.jsonl"
ASSETS = Path("assets"); ASSETS.mkdir(exist_ok=True)

def load_records():
    recs = []
    with open(LOGS, "r", encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except:
                pass
    return recs

def main():
    recs = load_records()
    if not recs:
        print("⚠️ No records found. Run training first.")
        return

    fig, ax = plt.subplots(2, 1, figsize=(7,6))
    ax1, ax2 = ax
    xs, reward, harmony = [], [], []
    affect, decision = [], []

    ln1, = ax1.plot([], [], label="Reward")
    ln2, = ax1.plot([], [], label="Harmony")
    ax1.legend(loc="lower right"); ax1.set_ylim(-1.1, 1.1)

    ln3, = ax2.plot([], [], label="Affect")
    ln4, = ax2.plot([], [], label="Decision")
    ax2.legend(loc="lower right"); ax2.set_ylim(-1.1, 1.1)

    def init():
        ax1.set_title("Learning Dynamics — Reward & Harmony")
        ax2.set_title("Alignment — Affect vs Decision")
        return ln1, ln2, ln3, ln4

    def update(i):
        xs.append(i)
        reward.append(recs[i]["reward"])
        harmony.append(recs[i]["harmony"])
        affect.append(recs[i]["affect"])
        decision.append(recs[i]["decision"])

        ln1.set_data(xs, reward)
        ln2.set_data(xs, harmony)
        ln3.set_data(xs, affect)
        ln4.set_data(xs, decision)

        ax1.set_xlim(0, max(50, i))
        ax2.set_xlim(0, max(50, i))
        return ln1, ln2, ln3, ln4

    ani = FuncAnimation(fig, update, frames=min(400, len(recs)), init_func=init, interval=30, blit=False)
    out_mp4 = ASSETS / "synmind_teaser.mp4"
    try:
        writer = FFMpegWriter(fps=30, bitrate=1800)
        ani.save(out_mp4, writer=writer)
        print("🎬 Saved:", out_mp4)
    except Exception as e:
        print("FFmpeg not available, saving GIF instead…", e)
        out_gif = ASSETS / "synmind_teaser.gif"
        ani.save(out_gif, writer="pillow", fps=20)
        print("🖼️ Saved:", out_gif)

if __name__ == "__main__":
    main()
