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

# ------------------------------
# Load PQ JSONs
# ------------------------------
with open("pq_comparison/eval_results_fcclip.json", "r") as f:
    data1 = json.load(f)
with open("pq_comparison/eval_results_tuned.json", "r") as f:
    data2 = json.load(f)
with open("pq_comparison/eval_results_reclip.json", "r") as f:
    data3 = json.load(f)

# Load overlap array (1=seen, 0=unseen)
with open("pq_comparison/train_overlap.json", "r") as f:
    overlap_array = json.load(f)

# Load GT file for weighting
with open("datasets/ADEChallengeData2016/ade20k_panoptic_val.json", "r") as f:
    gt_data = json.load(f)

# ------------------------------
# Compute per-class area from GT
# ------------------------------
class_areas = defaultdict(int)
for ann in gt_data["annotations"]:
    for seg in ann["segments_info"]:
        cid = str(seg["category_id"])
        class_areas[cid] += seg["area"]

# ------------------------------
# Extract PQs per class
# ------------------------------
pq1 = data1["pq_res"]["per_class"]  # Frozen
pq2 = data2["pq_res"]["per_class"]  # Tuned
pq3 = data3["pq_res"]["per_class"]  # ReCLIP

class_ids = sorted(set(pq1.keys()) | set(pq2.keys()) | set(pq3.keys()))

# ------------------------------
# Compute weighted PQ differences (Tuned - Frozen) for top-N selection
# ------------------------------
diffs_weighted = {cid: (pq2.get(cid, {"pq":0})["pq"] - pq1.get(cid, {"pq":0})["pq"]) * class_areas.get(cid, 0) for cid in class_ids}

# Separate seen/unseen
seen_classes = [cid for cid in class_ids if int(cid) < len(overlap_array) and overlap_array[int(cid)] == 1]
unseen_classes = [cid for cid in class_ids if int(cid) < len(overlap_array) and overlap_array[int(cid)] == 0]

# Top-N selection based on weighted difference
top_seen = sorted(seen_classes, key=lambda cid: abs(diffs_weighted[cid]), reverse=True)[:TOP_N_PER_GROUP]
top_unseen = sorted(unseen_classes, key=lambda cid: abs(diffs_weighted[cid]), reverse=True)[:TOP_N_PER_GROUP]

plot_classes = top_seen + top_unseen

# ------------------------------
# Prepare PQ values for plotting (%)
# ------------------------------
pq_values1 = [pq1.get(cid, {"pq":0})["pq"] * 100 for cid in plot_classes]  # Frozen
pq_values2 = [pq2.get(cid, {"pq":0})["pq"] * 100 for cid in plot_classes]  # Tuned
pq_values3 = [pq3.get(cid, {"pq":0})["pq"] * 100 for cid in plot_classes]  # ReCLIP

# Class names
class_names = [
    ADE20K_150_CATEGORIES[int(cid)]["name"].split(',')[0] if int(cid) < len(ADE20K_150_CATEGORIES) else f"Class {cid}"
    for cid in plot_classes
]

# ------------------------------
# Compute ReCLIP improvement over Frozen CLIP for averages
# ------------------------------
pq_delta_reclip = {cid: pq3.get(cid, {"pq":0})["pq"] - pq1.get(cid, {"pq":0})["pq"] for cid in class_ids}

# Weighted average
def weighted_avg_delta(class_list):
    total_weight, weighted_sum = 0, 0
    for cid in class_list:
        w = class_areas.get(cid, 0)
        delta = pq_delta_reclip.get(cid, 0)
        weighted_sum += delta * w
        total_weight += w
    return (weighted_sum / total_weight) * 100 if total_weight > 0 else 0

avg_seen_weighted = weighted_avg_delta(seen_classes)
avg_unseen_weighted = weighted_avg_delta(unseen_classes)

# Unweighted average
avg_seen_unweighted = np.mean([pq_delta_reclip[cid]*100 for cid in seen_classes])
avg_unseen_unweighted = np.mean([pq_delta_reclip[cid]*100 for cid in unseen_classes])

# ------------------------------
# Plotting
# ------------------------------
x = np.arange(len(plot_classes))
width = 0.25

fig, ax = plt.subplots(figsize=(25,6))
bars1 = ax.bar(x - width, pq_values1, width, label="Frozen CLIP")
bars2 = ax.bar(x, pq_values2, width, label="Tuned CLIP")
bars3 = ax.bar(x + width, pq_values3, width, label="ReCLIP")

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_xlabel("Class")
ax.set_ylabel("PQ (%)")
ax.set_title(
    f"PQ Comparison (Top {TOP_N_PER_GROUP} Seen & Unseen)\n"
    f"ReCLIP ΔPQ Unseen: {avg_unseen_unweighted:.2f}% (unweighted), {avg_unseen_weighted:.2f}% (weighted)\n"
    f"ReCLIP ΔPQ Seen: {avg_seen_unweighted:.2f}% (unweighted), {avg_seen_weighted:.2f}% (weighted)"
)
ax.legend()

# Separator between seen/unseen
ax.axvline(x=TOP_N_PER_GROUP - 0.5, color='gray', linestyle='--')
ax.text(TOP_N_PER_GROUP/2 - 2, max(max(pq_values1), max(pq_values2), max(pq_values3))*1.02 - 2, "Seen", ha='center', fontsize=12)
ax.text(TOP_N_PER_GROUP + TOP_N_PER_GROUP/2, max(max(pq_values1), max(pq_values2), max(pq_values3))*1.02 - 2, "Unseen", ha='center', fontsize=12)

plt.tight_layout()
plt.savefig("pq_difference_three_models.svg", format="svg")
plt.show()
