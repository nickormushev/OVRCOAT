import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
counter = {}

original = json.load(open('fcclip/tests/found_by_oracle/found-objects-original.json'))
oracle = json.load(open('fcclip/tests/found_by_oracle/found-objects-oracle.json'))

class Stat:
    def __init__(self):
        self.counter = 0
        self.total = 0

    def ratio(self):
        return self.counter / self.total

    def __str__(self):
        return '%.2f' % self.ratio()

    def __repr__(self):
        return '%.2f' % self.ratio()

img_diff_counter = {}

overlap = json.load(open('fcclip/tests/train_overlap_text.json'))

for img_id, obj in original.items():
    if img_id not in oracle:
        raise Exception('Object not found in oracle: %s' % obj)

    found_gt_original = obj
    found_gt_oracle = oracle[img_id]
    img_diff_counter[img_id] = abs(len(found_gt_original) - len(found_gt_oracle))
    
    for obj in found_gt_oracle:
        obj_name = obj[0]
        if obj_name in overlap:
            obj_name = f'*{obj_name.split(',')[0]}*'
        else:
            obj_name = obj_name.split(',')[0]

        val = obj[1]

        if obj_name not in counter:
            counter[obj_name] = Stat()
        counter[obj_name].total += 1

        if obj in found_gt_original:
            continue

        counter[obj_name].counter += 1

sorted_items = sorted(counter.items(), key=lambda item: item[1].ratio())
keys = [ key for key, val in sorted_items if val.ratio() > 0.4] 
values = [val.ratio() for key, val in sorted_items if val.ratio() > 0.4]


sns.set_theme(style="whitegrid")  # Applies the correct base style
sns.set_palette("muted") 

# ... [rest of your data processing code remains unchanged]

# Rebuild filtered + labeled data
keys = []
values = []
colors = []

for key, val in sorted_items:
    ratio = val.ratio()
    if ratio > 0.4:
        clean_name = key.replace('*', '')
        is_seen = key.startswith('*') and key.endswith('*')
        keys.append(clean_name)
        values.append(ratio)
        colors.append('tab:blue' if is_seen else 'tab:orange')

# Plot ratio (percentage missed)
plt.figure(figsize=(15, 6))
bars = plt.bar(keys, values, color=colors)

# Annotate bars
#y_max = max(values)
#for bar, val in zip(bars, values):
#    height = bar.get_height()
#    offset = y_max * 0.03
#    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
#             f'{val:.2f}', ha='center', va='bottom', fontsize=8)

# Axis labels and title
plt.xlabel('Category')
plt.ylabel('Missed Detection Rate')
plt.title('Objects Found by Oracle but Missed by FC-CLIP')
plt.xticks(rotation=45, ha='right')

# Legend
legend_elements = [Patch(facecolor='tab:blue', label='Seen category'),
                   Patch(facecolor='tab:orange', label='Unseen category')]
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('fcclip/tests/found_by_oracle/compare_improved.svg')
plt.close()

# Plot raw count (optional, same color code)
values_count = [val.counter for key, val in sorted_items if val.ratio() > 0.4]

plt.figure(figsize=(12, 6))
bars = plt.bar(keys, values_count, color=colors)

for bar, val in zip(bars, values_count):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val}', ha='center', va='bottom', fontsize=8)

plt.xlabel('Category')
plt.ylabel('Missed Count')
plt.title('Count of Missed Objects by Category (Seen vs Unseen)')
plt.xticks(rotation=45, ha='right')
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('fcclip/tests/found_by_oracle/missed_improved.png')
plt.close()
