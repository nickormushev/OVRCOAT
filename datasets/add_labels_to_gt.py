import json
import cv2
import os, sys

new_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(new_root_dir)
os.chdir(new_root_dir)
sys.path.append(new_root_dir)

from ovrcoat.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

pan_val = json.load(open('./datasets/ADEChallengeData2016/ade20k_panoptic_val.json'))


pan_anns = pan_val['annotations']

for ann in pan_anns:
    file = ann['image_id'] + ".png"
    gt_img = cv2.imread('./datasets/ADEChallengeData2016/ade20k_panoptic_val/' + file )

    for seg in ann['segments_info']:
        segment_name = ADE20K_150_CATEGORIES[seg['category_id']]['name'].split(',')[0]

        x, y, w, h = seg['bbox']
        center_x = x + w // 2
        center_y = y + h // 2

        # Put the segment name in the center of the bbox
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(segment_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2

        cv2.putText(gt_img, segment_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    cv2.imwrite('./datasets/ADEChallengeData2016/ade20k_panoptic_val_with_labels/' + file, gt_img)
        
