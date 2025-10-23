# dashboard/live_dashboard.py
import json, time, os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

LOGS = Path("logs")
STREAM_PATH = LOGS / "synmind_stream.jsonl"
ASSETS = Path("assets"); ASSETS.mkdir(exist_ok=True)

def tail_jsonl(path):
    """Generator: yields dicts as they appear in the JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        # jump to end then stream new lines
        f.seek(0, os.SEEK_SET)
        for line in f:
            try:
                yield json.loads(line)
            except:
                pass
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue
            try:
                yield json.loads(line)
            except:
                pass

def main():
    if not STREAM_PATH.exists():
        print("⚠️ Run:  python run_synmind.py  (in another terminal) — then start this dashboard.")
        return

    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    (ax1, ax2, ax3) = ax
    xs, reward, harmony, certainty = [], [], [], []
    affect, decision = [], []
    trust_e, trust_r = [], []

    ln1, = ax1.plot([], [], label="Reward")
    ln2, = ax1.plot([], [], label="Harmony")
    ln3, = ax1.plot([], [], label="Certainty")
    ax1.set_title("Reward / Harmony / Certainty"); ax1.legend(loc="lower right")

    ln4, = ax2.plot([], [], label="Affect")
    ln5, = ax2.plot([], [], label="Decision")
    ax2.set_title("Affect vs Decision"); ax2.legend(loc="lower right")

    ln6, = ax3.plot([], [], label="Trust Emotion")
    ln7, = ax3.plot([], [], label="Trust Reason")
    ax3.set_title("Trust Evolution"); ax3.legend(loc="lower right")

    stream = tail_jsonl(STREAM_PATH)

    def update(frame):
        try:
            rec = next(stream)
        except StopIteration:
            return ln1,
        idx = len(xs)
        xs.append(idx)
        reward.append(rec["reward"])
        harmony.append(rec["harmony"])
        certainty.append(rec["certainty"])
        affect.append(rec["affect"])
        decision.append(rec["decision"])
        trust_e.append(rec["trust_emotion"])
        trust_r.append(rec["trust_reason"])

        # update data
        ln1.set_data(xs, reward); ln2.set_data(xs, harmony); ln3.set_data(xs, certainty)
        ln4.set_data(xs, affect);  ln5.set_data(xs, decision)
        ln6.set_data(xs, trust_e); ln7.set_data(xs, trust_r)

        # rescale
        for a in (ax1, ax2, ax3):
            a.relim(); a.autoscale_view()

        return ln1, ln2, ln3, ln4, ln5, ln6, ln7

    ani = FuncAnimation(fig, update, interval=60, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()