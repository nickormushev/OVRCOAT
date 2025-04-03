"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
Reference: https://github.com/open-mmlab/mmdetection/pull/7538
"""

#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import wandb
import os, sys
import numpy as np
import json
import time
from datetime import timedelta
from collections import defaultdict
import argparse
import multiprocessing
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color

from detectron2.evaluation import COCOPanopticEvaluator

new_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(new_root_dir)
sys.path.append(new_root_dir)

from fcclip.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
import PIL.Image as Image

# If set to False we ignore missmatched classes for true positive calculations for PQ
CHECK_CLASSIFICATION = True
# If set to False we ignore VOID class if missclassified for the true postive calculations for PQ
CHECK_BACKGROUND = True


class COCOPanopticWandbEvaluator(COCOPanopticEvaluator):

    def evaluate(self):
        res = super().evaluate()
        #wandb.log(res['panoptic_seg'])
        return res

from panopticapi.utils import get_traceback, rgb2id

OFFSET = 256 * 256 * 256
VOID = 0

class PQStatCat():
        def __init__(self):
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

        def __iadd__(self, pq_stat_cat):
            self.iou += pq_stat_cat.iou
            self.tp += pq_stat_cat.tp
            self.fp += pq_stat_cat.fp
            self.fn += pq_stat_cat.fn
            return self

class PQStatObjectRecognition():
        def __init__(self):
            # GT not found
            self.not_found_objects_percent = 0.0
            self.not_found_objects = 0.0
            # Pred but no GT
            self.extra_objects_percent = 0.0
            self.extra_objects = 0.0
            # Misslabelled objects percent
            self.mislabeled_objects_percent = 0.0
            self.mislabeled_objects = 0.0
            # Object mistaken as background count
            self.object_mistaken_as_background_percent = 0.0
            self.object_mistaken_as_background = 0.0

            self.total_objects_gt = 0.0
            self.total_objects_pred = 0.0
            self.found_gt = []
        
        def calc_percentages(self):
            self.not_found_objects_percent = self.not_found_objects / self.total_objects_gt
            self.extra_objects_percent = self.extra_objects / self.total_objects_pred
            self.mislabeled_objects_percent = self.mislabeled_objects / self.total_objects_gt
            self.object_mistaken_as_background_percent = self.object_mistaken_as_background / self.total_objects_gt


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)
        self.obj_recogn_per_img = defaultdict(PQStatObjectRecognition)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        
        for img_id, obj_recogn in pq_stat.obj_recogn_per_img.items():
            self.obj_recogn_per_img[img_id] = obj_recogn
        
        return self
    
    def get_top_n_images_by_criteria(self, n: int, criteria: str = "missed"):
        if criteria == "missed":
            sorted_img_ids = sorted(self.obj_recogn_per_img.items(),
                                    key=lambda x: x[1].not_found_objects_percent, reverse=True)
        else:
            sorted_img_ids = sorted(self.obj_recogn_per_img.items(),
                                    key=lambda x: x[1].mislabeled_objects_percent, reverse=True)
        return sorted_img_ids[:n]
    
    def object_detection_percentage_info(self):
        img_count, not_found_total, mislabeled_as_background_total, mislabeled_total, extra_total  = 0, 0, 0, 0, 0
        not_found_percent, mislabeled_percent, mislabeled_as_background_percent, extra_percent = 0, 0, 0, 0
        total_obj_gt, total_obj_pred = 0, 0

        for _, info in self.obj_recogn_per_img.items():
            info.calc_percentages()
            # Per image percentages
            not_found_percent += info.not_found_objects_percent
            mislabeled_percent += info.mislabeled_objects_percent
            mislabeled_as_background_percent += info.object_mistaken_as_background
            extra_percent += info.extra_objects_percent

            # Total counts
            not_found_total += info.not_found_objects
            mislabeled_as_background_total += info.object_mistaken_as_background
            mislabeled_total += info.mislabeled_objects
            extra_total += info.extra_objects

            total_obj_gt += info.total_objects_gt
            total_obj_pred += info.total_objects_pred

            img_count += 1

        if img_count != 0:
            return { "Per_image": { 
                        "missed": not_found_percent / img_count, "misslabeled": mislabeled_percent/ img_count, 
                        "misslabeled_as_background": mislabeled_as_background_percent / img_count,
                        "extra": extra_percent / img_count, "img_count: ": img_count
                     },
                     "Total": {
                        "missed_objects": not_found_total / total_obj_gt, "misslabeled_objects": mislabeled_total/ total_obj_gt, 
                        "objects_misslabeled_as_background": mislabeled_as_background_total / total_obj_gt,
                        "extra_objects": extra_total / total_obj_pred, "gt_objects_count: ": total_obj_gt,
                        "pred_objects_count: ": total_obj_pred
                     }
                    }
        else:
            return {}


    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


# Annotation set holds a pair for pred and gt annotations
@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
    pq_stat = PQStat()
    pq_stat_seen = PQStat()
    pq_stat_unseen = PQStat()

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        # This flattens image making unique ids for each image
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        # Comment out if we are not using rgb version of images
        # pan_pred = rgb2id(pan_pred)

        pred_ann['segments_info'].append({'id': VOID, 'category_id': VOID, 'area': 1})

        # Make a map from segment id to segment info for both GT and prediction
        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        # Had issues adding directly to pred_segms
        pred_ann['segments_info'].pop()

        # Create set of all segment ids in the prediction
        # el['id'] gives the segment id which the pixels belonging to that segment
        # use as an identifier
        # used to validate if there are images that are in the segment info but not in the PNG
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])

        # Gets all unique labels and their counts. For each prediction they are unique
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)

        # For each label in the prediction validate it and caluclate area
        for label, label_cnt in zip(labels, labels_cnt):
            # Labels are from the image. We want to see if they are in the segment info
            # If not or they are void we skip or throw an error
            if label not in pred_segms or label == VOID:
                if label == VOID:
                    pred_segms[label]['area'] = label_cnt
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            
            # area is how many pixels have this label
            pred_segms[label]['area'] = label_cnt
            
            # Update if we have seen a label
            pred_labels_set.remove(label)

            # Check if the category_id is in categories taken from gt
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))

        # Check if there are any labels left in the set. Which means it is in annotation but not PNG
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

        # confusion matrix calculation
        # They make each pixel unique by multiplying the GT by OFFSET and adding the prediction
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            # The gt_id is the label // OFFSET which is the whole part based on how we built pan_gt_pred
            gt_id = label // OFFSET
            # Pred_id is the label % OFFSET which is the remainder part based on how we built pan_gt_pred
            pred_id = label % OFFSET

            # Intersection relies on the fact that each object has a unique id
            gt_pred_map[(gt_id, pred_id)] = intersection
        
        # count all matched pairs
        gt_matched = set()
        pred_matched = set()

        overlap = json.load(open("./tests/train_overlap.json", "r"))
        # For each pair of gt and pred
        missclassified_as_background_count = 0.0
        misslabeled = 0.0
        detected = 0.0
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple

            # More or less checks if gt_label is VOID
            # Checked and in other cases doesn't enter
            if gt_label not in gt_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
               continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union

            if iou > 0.5:
                detected += 1
                # If pred_label for a segment is VOID we skip
                # This tracks gt objects that exist but are classified as background
                if CHECK_BACKGROUND and pred_label == VOID:
                    missclassified_as_background_count += 1
                    # There was no mask at this location so object was not detected
                    detected -= 1
                    continue

                gt_category_id = gt_segms[gt_label]['category_id']
                CATEGORY_CLASSES = ADE20K_150_CATEGORIES
                pq_stat.obj_recogn_per_img[gt_ann['image_id']].found_gt += [(CATEGORY_CLASSES[gt_category_id]['name'], gt_category_id)] 

                if CHECK_CLASSIFICATION and gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
#                    print(f"GT: {CATEGORY_CLASSES[gt_segms[gt_label]['category_id']]['name']},   \
#Pred: {CATEGORY_CLASSES[pred_segms[pred_label]['category_id']]['name']}, File: {gt_ann['file_name']},\
#GT ID: {gt_label}, Pred ID: {pred_label}")
                    misslabeled += 1
                    continue

                if overlap[gt_category_id]:
                    pq_stat_seen[gt_segms[gt_label]['category_id']].tp += 1
                    pq_stat_seen[gt_segms[gt_label]['category_id']].iou += iou
                else:
                    pq_stat_unseen[gt_segms[gt_label]['category_id']].tp += 1
                    pq_stat_unseen[gt_segms[gt_label]['category_id']].iou += iou

                # If the category_id is not the same we skip. Not in my case
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)
        

        pq_stat.obj_recogn_per_img[gt_ann['image_id']].total_objects_gt = len(gt_segms) 
        pq_stat.obj_recogn_per_img[gt_ann['image_id']].object_mistaken_as_background = missclassified_as_background_count 
        pq_stat.obj_recogn_per_img[gt_ann['image_id']].mislabeled_objects = misslabeled
        pq_stat.obj_recogn_per_img[gt_ann['image_id']].not_found_objects = len(gt_segms) - detected

        # count false negatives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue

            # not found or not matched
            pq_stat[gt_info['category_id']].fn += 1

            if overlap[gt_info['category_id']]:
                pq_stat_seen[gt_info['category_id']].fn += 1
            else:
                pq_stat_unseen[gt_info['category_id']].fn += 1
        
        # count false positives
        extra_preds = 0.0
        pq_stat.obj_recogn_per_img[gt_ann['image_id']].total_objects_pred = len(pred_segms) 
        for pred_label, pred_info in pred_segms.items():
            # Case 1) It was matched
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)

            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)

            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                extra_preds += 1
                continue

            if overlap[pred_info['category_id']]:
                pq_stat_seen[pred_info['category_id']].fp += 1
            else:
                pq_stat_unseen[pred_info['category_id']].fp += 1

            pq_stat[pred_info['category_id']].fp += 1
        if len(pred_segms) != 0:
            pq_stat.obj_recogn_per_img[gt_ann['image_id']].extra_objects = extra_preds

    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat, pq_stat_unseen, pq_stat_seen


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories))
        processes.append(p)

    # https://github.com/open-mmlab/mmdetection/pull/7538
    # Close the process pool, otherwise it will lead to memory
    # leaking problems.
    workers.close()
    workers.join()


    pq_stat = PQStat()
    pq_stat_unseen = PQStat()
    pq_stat_seen = PQStat()
    for p in processes:
        pq_stat_proc, pq_stat_unseen_proc, pq_stat_seen_proc = p.get()
        pq_stat += pq_stat_proc
        pq_stat_unseen += pq_stat_unseen_proc
        pq_stat_seen += pq_stat_seen_proc
    return pq_stat, pq_stat_unseen, pq_stat_seen


def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            raise Exception('no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    pq_stat, pq_stat_unseen, pq_stat_seen = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)


    found_objects = { img_id: obj_recogn.found_gt for img_id, obj_recogn in pq_stat.obj_recogn_per_img.items()}

    json.dump(found_objects, open("found-objects-fixed.json", "w"))

    print("Per image panoptic quality metrics: ")
    object_detection_stats = pq_stat.object_detection_percentage_info()
    print("Percentages stats: ")
    print(json.dumps(object_detection_stats, indent=4))
    
    print("-" * 50)
    
    top_n = pq_stat.get_top_n_images_by_criteria(20, "misslabeled")
    for img in top_n:
        print(f"Image ID: {img[0]}, misslabeled percentage: {pq_stat.obj_recogn_per_img[img[0]].mislabeled_objects_percent}")

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    results_seen = {}
    results_unseen = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        results_seen[name], _ = pq_stat_seen.pq_average(categories, isthing=isthing)
        results_unseen[name], _ = pq_stat_unseen.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n'])
        )
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name + "_seen",
            100 * results_seen[name]['pq'],
            100 * results_seen[name]['sq'],
            100 * results_seen[name]['rq'],
            results_seen[name]['n'])
        )
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name + "_unseen",
            100 * results_unseen[name]['pq'],
            100 * results_unseen[name]['sq'],
            100 * results_unseen[name]['rq'],
            results_unseen[name]['n'])
        )


    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results, results_unseen, results_seen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground turth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--single-model',
                        action="store_true",
                        help="Single model used")

    parser.add_argument('--wandb-name',
                        default="TEST",
                        help="")

    args = parser.parse_args()

    wandb.init(
        name=args.wandb_name,
        project="segmentation-clip-detailed-no-norm",
    )

    print(os.getcwd())
    if args.single_model:
        pred_folder = f"{args.pred_folder}"
        pred_json_file = f"{pred_folder}/annotations.json"
        res, res_unseen, res_seen = pq_compute(args.gt_json_file, pred_json_file, args.gt_folder, pred_folder)
    else:
        for iter in ["0003999", "0007999", "0011999","0015999", "0019999"]:
            pred_folder = f"{args.pred_folder}{iter}"
            pred_json_file = f"{pred_folder}/annotations.json"
            res, res_unseen, res_seen = pq_compute(args.gt_json_file, pred_json_file, args.gt_folder, pred_folder)
            metrics = [("All", None), ("Things", True), ("Stuff", False)]

            for name, _isthing in metrics:
                wandb.log({
                    f"PQ_{name}": res[name]['pq'] * 100,
                    f"SQ_{name}": res[name]['sq'] * 100,
                    f"RQ_{name}": res[name]['rq'] * 100,

                    f"PQ_{name}_seen": res_seen[name]['pq'] * 100,
                    f"SQ_{name}_seen": res_seen[name]['sq'] * 100,
                    f"RQ_{name}_seen": res_seen[name]['rq'] * 100,

                    f"PQ_{name}_unseen": res_unseen[name]['pq'] * 100,
                    f"SQ_{name}_unseen": res_unseen[name]['sq'] * 100,
                    f"RQ_{name}_unseen": res_unseen[name]['rq'] * 100,
                }, step=int(iter))