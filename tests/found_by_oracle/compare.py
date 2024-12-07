import json
import matplotlib.pyplot as plt
counter = {}

original = json.load(open('tests/found_by_oracle/found-objects-original.json'))
oracle = json.load(open('tests/found_by_oracle/found-objects-oracle.json'))

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

for img_id, obj in original.items():
    if img_id not in oracle:
        raise Exception('Object not found in oracle: %s' % obj)

    found_gt_original = obj
    found_gt_oracle = oracle[img_id]
    img_diff_counter[img_id] = abs(len(found_gt_original) - len(found_gt_oracle))
    
    for obj in found_gt_oracle:
        obj_name = obj[0].split(',')[0]
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


plt.figure(figsize=(10, 5))
plt.bar(keys, values)
plt.xlabel('Category')
plt.ylabel('Percentage')
plt.title('What percent of elements in this category were found by oracle compared to fcclip')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('tests/found_by_oracle/compare.png')

values = [val.counter for key, val in sorted_items if val.ratio() > 0.4]

plt.figure(figsize=(10, 5))
plt.bar(keys, values)
plt.xlabel('Category')
plt.ylabel('Counts')
plt.title('How often this category was not found by oracle')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('tests/found_by_oracle/missed.png')

img_diff_counter_sorted = sorted(img_diff_counter.items(), key=lambda item: item[1])
json.dump(img_diff_counter_sorted, open('tests/found_by_oracle/img_diff_counter.json', 'w'), indent=4)