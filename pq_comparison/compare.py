import json
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from collections import defaultdict
from matplotlib.patheffects import withStroke

# ------------------------------
# User parameters
# ------------------------------
TOP_N_PER_GROUP = 10          # How many top classes per group
WEIGHT_SORTING = False         # Weight sorting & selection by mask count (True = weighted)
SHOW_MASK_COUNT = True        # Show mask counts on bars
FILTER_LOW_COUNTS = True      # Hide classes with very few masks
MIN_MASKS_FOR_DISPLAY = 10    # Minimum mask count to show class
OUTPUT_FILE = "pq_difference_side_by_side.svg"
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

# Load GT file for mask counts
with open("datasets/ADEChallengeData2016/ade20k_panoptic_val.json", "r") as f:
    gt_data = json.load(f)

# ------------------------------
# Compute per-class mask counts
# ------------------------------
class_counts = defaultdict(int)
for ann in gt_data["annotations"]:
    for seg in ann["segments_info"]:
        cid = str(seg["category_id"])
        class_counts[cid] += 1

# ------------------------------
# Extract PQs per class
# ------------------------------
pq1 = data1["pq_res"]["per_class"]
pq2 = data2["pq_res"]["per_class"]
pq3 = data3["pq_res"]["per_class"]
class_ids = sorted(set(pq1.keys()) | set(pq2.keys()) | set(pq3.keys()))

# ------------------------------
# Compute weighted PQ differences
# ------------------------------
diffs_weighted = {}
for cid in class_ids:
    diff = pq2.get(cid, {"pq": 0})["pq"] - pq1.get(cid, {"pq": 0})["pq"]
    weight = class_counts.get(cid, 1) if WEIGHT_SORTING else 1
    diffs_weighted[cid] = diff * weight

# ------------------------------
# Separate seen/unseen and filter
# ------------------------------
def filter_classes(class_list):
    if FILTER_LOW_COUNTS:
        class_list = [cid for cid in class_list if class_counts.get(cid, 0) >= MIN_MASKS_FOR_DISPLAY]
    return class_list

seen_classes = filter_classes([cid for cid in class_ids if int(cid) < len(overlap_array) and overlap_array[int(cid)] == 1])
unseen_classes = filter_classes([cid for cid in class_ids if int(cid) < len(overlap_array) and overlap_array[int(cid)] == 0])

# ------------------------------
# Top-N selection & sort by signed difference
# ------------------------------
def select_top_sorted(class_list):
    top_n = sorted(class_list, key=lambda cid: abs(diffs_weighted[cid]), reverse=True)[:TOP_N_PER_GROUP]
    return sorted(top_n, key=lambda cid: diffs_weighted[cid], reverse=True)

top_seen = select_top_sorted(seen_classes)
top_unseen = select_top_sorted(unseen_classes)

# ------------------------------
# Compute ReCLIP improvement over Frozen CLIP for averages
# ------------------------------
pq_delta_reclip = {cid: pq3.get(cid, {"pq":0})["pq"] - pq1.get(cid, {"pq":0})["pq"] for cid in class_ids}

def weighted_avg_delta(class_list):
    total_weight, weighted_sum = 0, 0
    for cid in class_list:
        w = class_counts.get(cid, 0)
        delta = pq_delta_reclip.get(cid, 0)
        weighted_sum += delta * w
        total_weight += w
    return (weighted_sum / total_weight) * 100 if total_weight > 0 else 0

avg_seen_weighted = weighted_avg_delta(seen_classes)
avg_unseen_weighted = weighted_avg_delta(unseen_classes)
avg_seen_unweighted = np.mean([pq_delta_reclip[cid]*100 for cid in seen_classes]) if seen_classes else 0
avg_unseen_unweighted = np.mean([pq_delta_reclip[cid]*100 for cid in unseen_classes]) if unseen_classes else 0

# ------------------------------
# Plot side-by-side

# ------------------------------
def plot_group(ax, class_ids_all, pq1, pq2, pq3, class_counts, top_n, title, show_mask_count=True):
    """
    Plots a group of classes (seen or unseen) sorted by delta PQ.
    Average delta is computed over all classes in the group.
    """
    # Compute signed delta (ReCLIP - Frozen)
    delta = lambda cid: pq3.get(cid, {"pq": 0})["pq"] - pq1.get(cid, {"pq": 0})["pq"]

    # Compute averages over all classes in the group
    deltas_all = [delta(cid) for cid in class_ids_all]
    weights_all = [class_counts.get(cid, 0) for cid in class_ids_all]
    avg_unweighted = np.mean([d * 100 for d in deltas_all]) if deltas_all else 0
    avg_weighted = (np.sum([d * w for d, w in zip(deltas_all, weights_all)]) / np.sum(weights_all) * 100) if weights_all else 0

    # Select top-N by absolute delta for plotting
    class_ids_top = sorted(class_ids_all, key=lambda cid: abs(delta(cid)), reverse=True)[:top_n]

    # Sort top classes by delta for plotting
    class_ids_sorted = sorted(class_ids_top, key=delta, reverse=True)

    # Prepare PQ values
    pq_vals1 = [pq1.get(cid, {"pq": 0})["pq"] * 100 for cid in class_ids_sorted]
    pq_vals2 = [pq2.get(cid, {"pq": 0})["pq"] * 100 for cid in class_ids_sorted]
    pq_vals3 = [pq3.get(cid, {"pq": 0})["pq"] * 100 for cid in class_ids_sorted]
    class_names = [
        ADE20K_150_CATEGORIES[int(cid)]["name"].split(",")[0] if int(cid) < len(ADE20K_150_CATEGORIES) else f"Class {cid}"
        for cid in class_ids_sorted
    ]

    x = np.arange(len(class_ids_sorted))
    width = 0.25

    # Bars
    ax.bar(x - width, pq_vals1, width, label="FC-CLIP")
    ax.bar(x, pq_vals2, width, label="Tuned CLIP")
    ax.bar(x + width, pq_vals3, width, label="ReCLIP")

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("PQ (%)", fontsize=12)
    ax.set_title(
        f"{title} (Top {top_n})\nΔ = PQ(ReCLIP) − PQ(Frozen CLIP)\n"
        f"ΔPQ: {avg_unweighted:.2f}% (unweighted), {avg_weighted:.2f}% (weighted)",
        fontsize=11, fontweight="bold", pad=15
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars
    for i, cid in enumerate(class_ids_sorted):
        d = delta(cid) * 100
        h = pq_vals3[i]
        n_masks = class_counts.get(cid, 0)
        text_y = h - 5 if h > 25 else h + 2
        va = "top" if h > 25 else "bottom"
        color = "#6ee16e" if d > 0.5 else "#ff6666" if d < -0.5 else "#f0f0f0"
        label = f"Δ={d:+.1f}%"
        if show_mask_count:
            label += f"\n(n={n_masks})"
        ax.text(
            i + width, text_y, label,
            ha="center", va=va, fontsize=9, color=color,
            fontweight="semibold",
            path_effects=[withStroke(linewidth=1.2, foreground="black", alpha=0.6)],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", edgecolor="none", alpha=0.5)
        )


# ------------------------------
# Create figure
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(30, 9),sharey=True)
plot_group(axes[0], seen_classes, pq1, pq2, pq3, class_counts, TOP_N_PER_GROUP, "Seen Classes")
plot_group(axes[1], unseen_classes, pq1, pq2, pq3, class_counts, TOP_N_PER_GROUP, "Unseen Classes")
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.savefig("pq_difference_seen_unseen.svg", format="svg")
plt.show()
