# Converts training overlap with ADE20K to text format
import json
import os, sys

new_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(new_root_dir)
os.chdir(new_root_dir)
sys.path.append(new_root_dir)

from fcclip.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

train_overlap = json.load(open('./tests/train_overlap.json'))

res = []
for idx, overlap in enumerate(train_overlap):
    if not overlap:
        res.append(ADE20K_150_CATEGORIES[idx]['name'])


json.dump(res, open('./tests/train_not_overlap_text.json', 'w'), indent=4)
        
