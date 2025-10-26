import json
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from collections import defaultdict

# ------------------------------
# User parameter: number of top classes to display per group
TOP_N_PER_GROUP = 10
# ------------------------------

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fcclip.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

# Load JSONs
with open("pq_comparison/eval_results_fcclip.json", "r") as f:
    data1 = json.load(f)

with open("pq_comparison/eval_results_tuned.json", "r") as f:
    data2 = json.load(f)

with open("pq_comparison/train_overlap.json", "r") as f:
    overlap_array = json.load(f)

with open("datasets/ADEChallengeData2016/ade20k_panoptic_val.json", "r") as f:
    gt_data = json.load(f)

# ---------------------------------------------
# Compute per-class area and count from GT
# ---------------------------------------------
class_areas = defaultdict(int)
class_counts = defaultdict(int)

for ann in gt_data["annotations"]:
    for seg in ann["segments_info"]:
        cid = str(seg["category_id"])
        class_areas[cid] += seg["area"]
        class_counts[cid] += 1

# ---------------------------------------------
# Extract PQs
# ---------------------------------------------
pq1 = data1["pq_res"]["per_class"]
pq2 = data2["pq_res"]["per_class"]
class_ids = sorted(set(pq1.keys()) | set(pq2.keys()))

# Compute difference and weighted averages
diffs = {cid: pq2.get(cid, {"pq":0})["pq"] - pq1.get(cid, {"pq":0})["pq"] for cid in class_ids}

# Separate seen/unseen based on overlap array
seen_classes = [cid for cid in class_ids if int(cid) < len(overlap_array) and overlap_array[int(cid)] == 1]
unseen_classes = [cid for cid in class_ids if int(cid) < len(overlap_array) and overlap_array[int(cid)] == 0]

# Weighted average by area
def weighted_avg(class_list):
    total_weight, weighted_sum = 0, 0
    for cid in class_list:
        w = class_areas.get(cid, 0)
        weighted_sum += diffs[cid] * w
        total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0

avg_seen_weighted = weighted_avg(seen_classes)
avg_unseen_weighted = weighted_avg(unseen_classes)

# Unweighted average
avg_seen_unweighted = np.mean([diffs[cid] for cid in seen_classes])
avg_unseen_unweighted = np.mean([diffs[cid] for cid in unseen_classes])

# ---------------------------------------------
# Select top-N per group for plotting
# ---------------------------------------------
weighted_diffs = {cid: diffs[cid] * class_areas.get(cid, 0) for cid in class_ids}

top_seen = sorted(seen_classes, key=lambda cid: abs(diffs[cid]), reverse=True)[:TOP_N_PER_GROUP]
top_unseen = sorted(unseen_classes, key=lambda cid: abs(diffs[cid]), reverse=True)[:TOP_N_PER_GROUP]
top_seen = sorted(seen_classes, key=lambda cid: abs(weighted_diffs[cid]), reverse=True)[:TOP_N_PER_GROUP]
top_unseen = sorted(unseen_classes, key=lambda cid: abs(weighted_diffs[cid]), reverse=True)[:TOP_N_PER_GROUP]

plot_classes = top_seen + top_unseen

# ---------------------------------------------
# Prepare PQ values for plotting (×100 for %)
# ---------------------------------------------
pq_values1 = [pq1.get(cid, {"pq":0})["pq"] * 100 for cid in plot_classes]
pq_values2 = [pq2.get(cid, {"pq":0})["pq"] * 100 for cid in plot_classes]

# Get readable names
class_names = [
    ADE20K_150_CATEGORIES[int(cid)]["name"].split(',')[0]
    if int(cid) < len(ADE20K_150_CATEGORIES)
    else f"Class {cid}"
    for cid in plot_classes
]

# ---------------------------------------------
# Plot
# ---------------------------------------------
x = np.arange(len(plot_classes))
width = 0.35

fig, ax = plt.subplots(figsize=(25,6))
bars1 = ax.bar(x - width/2, pq_values1, width, label="Frozen CLIP")
bars2 = ax.bar(x + width/2, pq_values2, width, label="Tuned CLIP")

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_xlabel("Class")
ax.set_ylabel("PQ (%)")
ax.set_title(
    f"PQ Comparison (Top {TOP_N_PER_GROUP} Seen & Unseen)\n"
    f"Unseen mean(ΔPQ): {avg_unseen_unweighted*100:.2f}% (unweighted), {avg_unseen_weighted*100:.2f}% (weighted)\n"
    f"Seen mean(ΔPQ): {avg_seen_unweighted*100:.2f}% (unweighted), {avg_seen_weighted*100:.2f}% (weighted)"
)
ax.legend()

# Separator between seen/unseen
ax.axvline(x=TOP_N_PER_GROUP - 0.5, color='gray', linestyle='--')
ax.text(TOP_N_PER_GROUP/2 - 2, max(pq_values2)*1.02 - 2, "Seen", ha='center', fontsize=12)
ax.text(TOP_N_PER_GROUP + TOP_N_PER_GROUP/2, max(pq_values2)*1.02 - 2, "Unseen", ha='center', fontsize=12)

plt.tight_layout()
plt.savefig("pq_difference_weighted.svg", format="svg")
plt.show()
