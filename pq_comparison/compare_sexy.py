import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke
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
FONT_SIZE = 22
OUTPUT_FILE = "pq_difference_side_by_side.pdf"
# ------------------------------

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ovrcoat.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

# ------------------------------
# Load PQ JSONs
# ------------------------------
with open("pq_comparison/eval_results_fcclip.json", "r") as f:
    data1 = json.load(f)
with open("pq_comparison/eval_results_tuned_new.json", "r") as f:
    data2 = json.load(f)
with open("pq_comparison/eval_results_reclip_new.json", "r") as f:
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
# Separate seen/unseen and filter
# ------------------------------
def filter_classes(class_list):
    if FILTER_LOW_COUNTS:
        class_list = [cid for cid in class_list if class_counts.get(cid, 0) >= MIN_MASKS_FOR_DISPLAY]
    return class_list

seen_classes = filter_classes([cid for cid in class_ids if int(cid) < len(overlap_array) and overlap_array[int(cid)] == 1])
unseen_classes = filter_classes([cid for cid in class_ids if int(cid) < len(overlap_array) and overlap_array[int(cid)] == 0])


# ------------------------------
# Sexy color palette (colorblind friendly)
# ------------------------------
# Updated flat color palette
COLORS = {
    "FC-CLIP": "#2C3E50",       # flat blue
    "BASELINE+OVR": "#E74C3C",  # coral
    "OVRCOAT": "#16A085"        # soft green
}

# Enhanced plotting function with solid black annotation boxes
def plot_group(ax, class_ids_all, pq1, pq2, pq3, class_counts, top_n, title, show_mask_count=True, offsets=False):
    delta = lambda cid: pq3.get(cid, {"pq":0})["pq"] - pq1.get(cid, {"pq":0})["pq"]
    class_ids_top = sorted(class_ids_all, key=lambda cid: abs(delta(cid)), reverse=True)[:top_n]
    class_ids_sorted = sorted(class_ids_top, key=delta, reverse=True)

    # PQ values
    pq_vals1 = [pq1.get(cid, {"pq":0})["pq"]*100 for cid in class_ids_sorted]
    pq_vals2 = [pq2.get(cid, {"pq":0})["pq"]*100 for cid in class_ids_sorted]
    pq_vals3 = [pq3.get(cid, {"pq":0})["pq"]*100 for cid in class_ids_sorted]
    name_substitutions = {
        "apparel": "clothes",
        "plant pots": "pot",
        "television receiver": "tv",
        "bathtub": "tub",
        "ceiling fan": "fan",
        "ashcan": "trash can"
    }
    class_names = [
        name_substitutions.get(ADE20K_150_CATEGORIES[int(cid)]["name"].split(",")[0], ADE20K_150_CATEGORIES[int(cid)]["name"].split(",")[0])
        if int(cid) < len(ADE20K_150_CATEGORIES) else f"Class {cid}"
        for cid in class_ids_sorted
    ]

    

    x = np.arange(len(class_ids_sorted))
    width = 0.25

    # Draw bars
    bar1 = ax.bar(x - width, pq_vals1, width, label="BASELINE(FC-CLIP)", color=COLORS["FC-CLIP"], edgecolor="black", alpha=0.9)
    bar2 = ax.bar(x, pq_vals2, width, label="BASELINE+OVR", color=COLORS["BASELINE+OVR"], edgecolor="black", alpha=0.9)
    bar3 = ax.bar(x + width, pq_vals3, width, label="OVRCOAT", color=COLORS["OVRCOAT"], edgecolor="black", alpha=0.9)


    # Axes styling
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=FONT_SIZE)
    ax.set_ylabel("PQ (%)", fontsize=FONT_SIZE)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_facecolor("#FFFFFF")  # white background for clarity

    # Compute signed delta (ReCLIP - Frozen)
    delta = lambda cid: pq3.get(cid, {"pq": 0})["pq"] - pq1.get(cid, {"pq": 0})["pq"]

    # Compute averages over all classes in the group
    deltas_all = [delta(cid) for cid in class_ids_all]
    weights_all = [class_counts.get(cid, 0) for cid in class_ids_all]
    # Title inside axes (LaTeX style, bold box)
    avg_unweighted = np.mean([delta(cid)*100 for cid in class_ids_all]) if class_ids_all else 0
    avg_weighted = (np.sum([d * w for d, w in zip(deltas_all, weights_all)]) / np.sum(weights_all) * 100) if weights_all else 0
    title_str = (
        f"$\mathrm{{mean}}(\Delta \mathrm{{PQ}}_{{{title}}}): {avg_weighted:.2f} \mathrm{{pp}}$"
    )
    ax.text(
        0.01, 0.98, title_str,
        transform=ax.transAxes,
        fontsize=FONT_SIZE + 2,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="left",
    )

    if "Unseen" == title:
        info_str = (
            r"$\Delta = \mathrm{PQ}^{\mathrm{OVRCOAT}} - \mathrm{PQ}^{\mathrm{FC\text{-}CLIP}}$" "\n"
            r"$\mathrm{{n}} = \text{mask count in ADE20K val}$" "\n"
        )

        ax.text(
            0.62, 0.80, info_str,
            transform=ax.transAxes,
            fontsize=FONT_SIZE + 2,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
        )

    # Annotate each bar with Δ labels
    for i, cid in enumerate(class_ids_sorted):
        d = delta(cid)*100
        h = pq_vals3[i]
        n_masks = class_counts.get(cid, 0)
        text_y = h - 5 if h > 25 else h + 2
        va = "top" if h > 25 else "bottom"

        # Δ text color: green=positive, red=negative, gray=neutral
        txt_color = "#2ECC71" if d > 0 else "#E67E22" if d < 0 else "#444444"
        label = f"Δ={d:+.1f} pp"
        if show_mask_count:
            label += f"\n(n={n_masks})"

        # Optional manual offsets for readability
        if offsets:
            if i == 2: text_y = h - 2
            if i == 3: text_y = h - 6
            if i == 4: text_y = h + 6
            if i == 5: text_y = h - 5
            if i == 6: text_y = h - 4
            if i == 8: text_y = h + 4
        else:
            if i == 1: text_y = h - 5
            if i == 7: text_y = h + 2

        ax.text(
            i + width - 0.3, text_y, label,
            ha="center", va=va, fontsize=FONT_SIZE - 2,

            # Color-coded Δ text
            color=txt_color,
            fontweight="bold",

            # Black background box for maximal contrast
            #bbox=dict(boxstyle="round,pad=0.3", facecolor="black", edgecolor="none", alpha=0.75),
            bbox=dict(
                boxstyle="round,pad=0.3,rounding_size=0.3",
                facecolor="#2C3E50",
                edgecolor="black",
                linewidth=1,    # <-- added a small border width
                alpha=1.0  # previously 0.8 → more transparent
            ),

            # Subtle stroke for better readability
            path_effects=[withStroke(linewidth=1.2, foreground="black", alpha=0.8)]
        )

    # Strengthen bar edges for style
    for bar_group in [bar1, bar2, bar3]:
        for bar in bar_group:
            bar.set_linewidth(1.5)
            bar.set_edgecolor("#333333")

# ------------------------------
# Create figure
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(30, 11), sharey=True)
# Place legend neatly
plot_group(axes[0], seen_classes, pq1, pq2, pq3, class_counts, TOP_N_PER_GROUP, "Seen")
plot_group(axes[1], unseen_classes, pq1, pq2, pq3, class_counts, TOP_N_PER_GROUP, "Unseen", offsets=True)


axes[1].legend(fontsize=FONT_SIZE, loc="upper right")  # no bbox_to_anchor
# Final layout
plt.tight_layout()
plt.savefig(OUTPUT_FILE, format="pdf")
plt.show()
