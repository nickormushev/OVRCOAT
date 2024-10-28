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
import os, sys
import numpy as np
import json
import time
from datetime import timedelta
from collections import defaultdict
import argparse
import multiprocessing
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color

import PIL.Image as Image

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
            # Pred but no GT
            self.extra_objects_percent = 0.0


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
    
    def object_detection_percentage_info(self):
        count, not_found, extra  = 0, 0, 0
        for _, info in self.obj_recogn_per_img.items():
            not_found += info.not_found_objects_percent
            extra += info.extra_objects_percent
            count += 1

        if count != 0:
            return {"missed": not_found / count, "extra": extra / count, "count: ": count}
        else:
            return {"missed": 0, "extra": 0, "count: ": 0}


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

    idx = 0
    # Walk through all images
    for gt_ann, pred_ann in annotation_set:
        # Each 100 images print the progress
        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        # Load GT and prediction panoptic segmentation for one image
        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        # This flattens image making unique ids for each image
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        # Comment out if we are not using rgb version of images
        # pan_pred = rgb2id(pan_pred)

        # Make a map from segment id to segment info for both GT and prediction
        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        # predicted segments area calculation + prediction sanity checks

        # Create set of all segment ids in the prediction
        # el['id'] gives the segment id which the pixels belonging to that segment
        # use as an identifier
        # used to validate if there are images that are in the segment info but not in the PNG
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])

        # Gets all unique labels and their counts. For each prediction they are unique
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)

        # For each label in the prediction validate it
        for label, label_cnt in zip(labels, labels_cnt):
            # Labels are from the image. We want to see if they are in the segment info
            # If not or they are void we skip or throw an error
            if label not in pred_segms:
                if label == VOID:
                    # pred_segms[label]['area'] = label_cnt
                    # If I find an error try this !!!!!!
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
        
        # Idea: Look at the interesections such that pred_id = 0 and gt_id != 0 and the opposite
        # I can count them and see how often that happens and print that out as a metric
        # Do IoU maybe and check if over 0.5
        # TP is if pred_id != 0 and gt_id != 0 and IoU > 0.5
        # TP is if pred_id == 0 and gt_id == 0 and IoU > 0.5
        # TN otherwise
        # Realistically I care more for how often the model is wrong than right
        # So maybe we can look at the FP and FN where pred_id != 0 and gt_id == 0 and the opposite . Also the IoU > 0.5 still

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()

        # For each pair of gt and pred
        mislabeled = []
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union


            if iou > 0.5:
                if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                    mislabeled += [gt_label]
                    continue
                # If the category_id is not the same we skip. Not in my case
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false negatives
        crowd_labels_dict = {}
        missed_obj = 0.0
        total_obj = len(gt_segms)
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue

            # not found
            if gt_label not in mislabeled:
                missed_obj += 1

            # not found or not matched
            pq_stat[gt_info['category_id']].fn += 1
        
        if total_obj != 0:
            pq_stat.obj_recogn_per_img[gt_ann['image_id']].not_found_objects_percent = missed_obj / total_obj

        # count false positives
        extra_preds = 0.0
        total_obj = len(pred_segms)
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

            pq_stat[pred_info['category_id']].fp += 1
        if total_obj != 0:
            pq_stat.obj_recogn_per_img[gt_ann['image_id']].extra_objects_percent = extra_preds / total_obj

    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat


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
    for p in processes:
        pq_stat += p.get()
    return pq_stat


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

    pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)

    print("Per image panoptic quality metrics: ")
    print("Missed Percentages: ", pq_stat.object_detection_percentage_info())

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
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


    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


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
    args = parser.parse_args()

    print(os.getcwd())
    pq_compute(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder)