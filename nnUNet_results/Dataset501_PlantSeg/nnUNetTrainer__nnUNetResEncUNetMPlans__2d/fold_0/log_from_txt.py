import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- CONFIG ----------
log_file = "training_log_2025_11_13_11_42_19.txt"  # adjust if needed
output_folder = "."
# ----------------------------

# robust float pattern: optional sign, decimals, optional exponent
NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

metrics = {}
current_epoch = None

with open(log_file, "r") as f:
    for line in f:
        # detect epoch
        m = re.search(r"Epoch\s+(\d+)", line)
        if m:
            current_epoch = int(m.group(1))
            metrics.setdefault(current_epoch, {})
            continue

        if current_epoch is None:
            continue

        # learning rate
        m = re.search(r"Current learning rate:\s*" + NUM, line)
        if m:
            metrics[current_epoch]["lr"] = float(m.group(1))

        # train loss (may be negative)
        m = re.search(r"train_loss\s*" + NUM, line)
        if m:
            metrics[current_epoch]["train_loss"] = float(m.group(1))

        # val loss (may be negative)
        m = re.search(r"val_loss\s*" + NUM, line)
        if m:
            metrics[current_epoch]["val_loss"] = float(m.group(1))

        # pseudo dice (usually >=0 but we keep same pattern)
        m = re.search(r"Pseudo dice \[np\.float32\(" + NUM + r"\)\]", line)
        if m:
            metrics[current_epoch]["mean_fg_dice"] = float(m.group(1))

        # epoch time
        m = re.search(r"Epoch time:\s*" + NUM + r"\s*s", line)
        if m:
            metrics[current_epoch]["epoch_time"] = float(m.group(1))

# ---------- build arrays: all epochs must be present ----------
epochs = sorted(metrics.keys())
print("Epochs found:", epochs[0], "to", epochs[-1], "| count:", len(epochs))

# if you expect exactly 0..139, enforce it:
if epochs[0] != 0 or epochs[-1] != 139 or len(epochs) != 140:
    raise RuntimeError("Expected epochs 0..139, got: "
                       f"{epochs[0]}..{epochs[-1]} (n={len(epochs)})")

train = np.array([metrics[e]["train_loss"] for e in epochs])
val   = np.array([metrics[e]["val_loss"]   for e in epochs])
dice  = np.array([metrics[e]["mean_fg_dice"] for e in epochs])
time  = np.array([metrics[e]["epoch_time"] for e in epochs])
lr    = np.array([metrics[e]["lr"]        for e in epochs])

print("Example train losses (can be negative):", train[:10])

# ---------- EMA pseudo-dice like nnUNetLogger ----------
ema = []
for i, v in enumerate(dice):
    if i == 0:
        ema.append(v)
    else:
        ema.append(ema[-1] * 0.9 + 0.1 * v)
ema = np.array(ema)

# ================== PLOTTING (continuous, bold, big) ==================
# ================== PLOTTING (continuous, bold, big, WHITE BACKGROUND) ==================
# ================== STYLING FIX → WHITE BACKGROUND + BLACK GRID ==================
sns.set(font_scale=4.2)
sns.set_style("white")

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.color'] = 'grey'
plt.rcParams['grid.linewidth'] = 1.2

label_fs  = 48
tick_fs   = 34
legend_fs = 34
label_font = dict(size=label_fs, weight='bold')

fig, ax_all = plt.subplots(3, 1, figsize=(34, 62))
x = epochs


# ---------------- TOP PANEL ----------------
ax = ax_all[0]
ax2 = ax.twinx()

ax.plot(x, train, color='b', linewidth=8, label="loss_tr")
ax.plot(x, val,   color='r', linewidth=8, label="loss_val")
ax2.plot(x, dice, color='g', ls='dotted', linewidth=7, label="pseudo dice")
ax2.plot(x, ema,  color='g', linewidth=8, label="pseudo dice (mov. avg.)")

ax.set_xlabel("epoch", fontdict=label_font)
ax.set_ylabel("loss", fontdict=label_font)
ax2.set_ylabel("pseudo dice", fontdict=label_font)

ax.tick_params(labelsize=tick_fs)
ax2.tick_params(labelsize=tick_fs)

ax.legend(loc=(0, 1), fontsize=legend_fs)
ax2.legend(loc=(0.2, 1), fontsize=legend_fs)

ax.grid(True)      # << BLACK GRID VISIBLE
# ax2.grid(False)  # optional to avoid double overlay


# ---------------- MIDDLE PANEL ----------------
ax = ax_all[1]
ax.plot(x, time, color='b', linewidth=8, label="epoch duration")
ax.set_ylim(0, ax.get_ylim()[1])

ax.set_xlabel("epoch", fontdict=label_font)
ax.set_ylabel("time [s]", fontdict=label_font)

ax.tick_params(labelsize=tick_fs)
ax.legend(loc=(0, 1), fontsize=legend_fs)
ax.grid(True)      # << GRID HERE TOO


# ---------------- BOTTOM PANEL ----------------
ax = ax_all[2]
ax.plot(x, lr, color='b', linewidth=8, label="learning rate")

ax.set_xlabel("epoch", fontdict=label_font)
ax.set_ylabel("learning rate", fontdict=label_font)

ax.tick_params(labelsize=tick_fs)
ax.legend(loc=(0, 1), fontsize=legend_fs)
ax.grid(True)      # << GRID HERE TOO

plt.tight_layout()
fig.savefig("progress_WHITE_GRID1.png", dpi=350, facecolor="white")
plt.close()

