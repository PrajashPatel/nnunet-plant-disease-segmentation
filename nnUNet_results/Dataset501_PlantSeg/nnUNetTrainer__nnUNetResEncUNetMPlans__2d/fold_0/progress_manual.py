import re

log_file = "training_log_2025_11_13_11_42_19.txt"

epochs_found = set()
with open(log_file) as f:
    for line in f:
        m = re.search(r"Epoch\s+(\d+)", line)
        if m:
            epochs_found.add(int(m.group(1)))

print("min epoch:", min(epochs_found))
print("max epoch:", max(epochs_found))
print("count    :", len(epochs_found))
