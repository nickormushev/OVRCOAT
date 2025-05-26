import json


with open('./datasets/ADEChallengeData2016/ade20k_panoptic_train_real.json', 'r') as f:
    val_data = json.load(f)


val_data['images'] = val_data['images'][0:5000]
val_data['annotations'] = val_data['annotations'][0:5000]

json.dump(val_data, open('./datasets/ADEChallengeData2016/ade20k_panoptic_val_5000.json', 'w'), indent=4)