import pandas as pd
from detectron2.config.config import CfgNode as CN
import sys
import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm 
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt
from collections import defaultdict
from fcclip.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES, ADE20k_COLORS

import multiprocessing as mp
from detectron2.utils.visualizer import random_color
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog
)

IDX_TO_CLASS = []

from detectron2.engine.defaults import DefaultPredictor as d2_defaultPredictor
from panopticapi.utils import rgb2id

class TestConfig:
    def __init__(self):
        self.skip_seen_files = False
        self.use_extended_categories = False
        self.save_pan_predictions = True
        self.use_colors = False

        # TODO: Rename to run_tests
        # all of the below require run_tests to be true / use oracle
        self.use_oracle = True

        self.use_clip_oracle = False
        self.use_class_oracle = False

        self.calculate_confusion_matrix = False
        self.calculate_void_clip_classifications = False

        # Metrics from above ran only for masks from hungarian matching
        # require class oracle
        self.calculate_confusion_matrix_best = False
        self.calculate_void_clip_classifications_best = False

        self.void_histogram_data = False

        self.evaluate = False
        # highlight_missed requires self.evaluate
        self.highlight_missed = False

class DefaultPredictor(d2_defaultPredictor):
    def set_metadata(self, metadata):
        self.model.set_metadata(metadata)

# TODO: Research if there is a better way to do this cause this is a bit insane
new_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(new_root_dir)
sys.path.append(new_root_dir)

from fcclip.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from datasets.prepare_ade20k_full_sem_seg import ADE20K_SEM_SEG_FULL_CATEGORIES
from fcclip import add_maskformer2_config, add_fcclip_config
from fcclip import MaskFormerPanopticDatasetMapper
 
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.MODEL.BACKBONE.FREEZE = True
    cfg.TEST.PERFECT_MASKS = False
    cfg.TEST.WITH_VOID = False
    cfg.TEST.WITH_FC_CLIP = False
    cfg.TRAIN = CN()
    cfg.TRAIN.SEG_HEAD = False
    cfg.TRAIN.WITH_FC_CLIP_MASKS = False
    cfg.TRAIN.LOSSES = ["masks", "labels", "oov_ce"]
    cfg.TRAIN.USE_TUNED_FEATURES_FOR_SEG_HEAD = False
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="fcclip demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory where input images are"
    )
    parser.add_argument(
        "--output-dir",
        default="./tests/preds-1",
        help="A directory to save outputs"
    )

    parser.add_argument(
        "--annotations_file_name",
        default="annotations.json",
        help="A directory to save outputs"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="modify config options using the command-line 'key value' pairs",
        # hardcoded default
        default=["MODEL.WEIGHTS", "/home/nikolay/Downloads/fcclip_cocopan.pth"],
        nargs=argparse.REMAINDER,
    )

    return parser


def extend_open_vocab_dataset():
    info = pd.read_csv("/home/nikolay/fc-clip-fork/datasets/extend_object_info_one_word.csv")
    stuff_classes = ["" for _ in range(len(ADE20K_150_CATEGORIES))]

    # Save the DataFrame to a CSV file
    for i,row in enumerate(ADE20K_150_CATEGORIES):
        for extended_row in info['name']:
            if extended_row.split(",")[0] == row['name'].split(",")[0]:
                stuff_classes[i] = f"{extended_row}"


    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(ADE20K_150_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i


    DatasetCatalog.register(
        "openvocab_dataset", lambda x: []
    )

    return MetadataCatalog.get("openvocab_dataset").set(
        stuff_classes=stuff_classes,
        stuff_colors=ADE20k_COLORS[:],
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id
    )

# Dataset that combines labels from COCO, ADE20K and LVIS
def create_open_vocab_dataset():
    coco_metadata = MetadataCatalog.get("openvocab_coco_2017_val_panoptic_with_sem_seg")
    ade20k_metadata = MetadataCatalog.get("openvocab_ade20k_panoptic_val")
    lvis_classes = open("./fcclip/data/datasets/lvis_1203_with_prompt_eng.txt", 'r').read().splitlines()
    lvis_classes = [x[x.find(':')+1:] for x in lvis_classes]
    lvis_colors = list(
        itertools.islice(itertools.cycle(coco_metadata.stuff_colors), len(lvis_classes))
    )
    # rerrange to thing_classes, stuff_classes
    coco_thing_classes = coco_metadata.thing_classes
    coco_stuff_classes = [x for x in coco_metadata.stuff_classes if x not in coco_thing_classes]
    coco_thing_colors = coco_metadata.thing_colors
    coco_stuff_colors = [x for x in coco_metadata.stuff_colors if x not in coco_thing_colors]
    ade20k_thing_classes = ade20k_metadata.thing_classes
    ade20k_stuff_classes = [x for x in ade20k_metadata.stuff_classes if x not in ade20k_thing_classes]
    ade20k_thing_colors = ade20k_metadata.thing_colors
    ade20k_stuff_colors = [x for x in ade20k_metadata.stuff_colors if x not in ade20k_thing_colors]

    user_classes = []
    user_colors = [random_color(rgb=True, maximum=1) for _ in range(len(user_classes))]

    # Adding all the classes affects the results
    stuff_classes = coco_stuff_classes + ade20k_stuff_classes
    stuff_colors = coco_stuff_colors + ade20k_stuff_colors
    thing_classes = user_classes + coco_thing_classes + ade20k_thing_classes + lvis_classes
    thing_colors = user_colors + coco_thing_colors + ade20k_thing_colors + lvis_colors

    thing_dataset_id_to_contiguous_id = {x: x for x in range(len(thing_classes))}
    DatasetCatalog.register(
        "openvocab_dataset", lambda x: []
    )

    return MetadataCatalog.get("openvocab_dataset").set(
        stuff_classes=thing_classes+stuff_classes,
        stuff_colors=thing_colors+stuff_colors,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
    )

def get_color_palette(num_colors):
    np.random.seed(42)  # For reproducibility
    return np.random.randint(0, 255, size=(num_colors + 1, 3), dtype=np.uint8)

def get_segment_index_by_id(id, list):
    if id == 0:
        return -1

    for i, item in enumerate(list):
        if item['id'] == id:
            return i
    
    raise Exception("Segment not found")

def process_segment(mask, uid, si, segment_idx, text_mask):
    y, x = np.where(mask)
    if len(y) > 0 and len(x) > 0:
        if uid != 0:
            category_id = si[segment_idx]['category_id']
            #isthing = si[segment_idx]['isthing']
            category_name = IDX_TO_CLASS[category_id].split(",")[0]
        else:
            category_name = "empty"
        centroid_y = int(np.mean(y))
        centroid_x = int(np.mean(x))
        font_scale = 0.4  # Increase font size
        font_thickness = 1  # Make text more bold
        cv2.putText(text_mask, category_name , (centroid_x, centroid_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

def apply_color_palette(segmentation, palette, dict):
    if len(palette) == 0:
        return segmentation
    h, w = segmentation.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(segmentation)
    text_mask = np.zeros((h, w, 3), dtype=np.uint8)  # Separate mask for text
    si = dict["segments_info"]
    for uid in unique_ids:
        mask = segmentation == uid
        colored_mask[mask] = palette[uid % len(palette)]
        segment_idx = get_segment_index_by_id(uid, si)
        process_segment(mask, uid, si, segment_idx, text_mask)
        if uid != 0:
            si[segment_idx]['rgb2id'] = rgb2id(palette[uid % len(palette)].tolist())

    combined_mask = cv2.addWeighted(colored_mask, 1, text_mask, 1, 0)
    dict["segments_info"] = si
    return combined_mask, dict


def process_image(predictor, img_path, img_file, output_dir, pan_annotations, test_cfg):
    img = read_image(img_path, format="BGR")

    # Add gt to predictor before calling it and pass it inside of the 
    # predictor to the model
    img_id = img_file.split(".")[0]
    predictor.test_cfg = test_cfg
    predictor.gt_img_id = img_id

    pred = predictor(img)

    if not test_cfg.save_pan_predictions:
        return

    dict = {
        "image_id": img_id,
        "file_name": img_id + ".png",
        # Segment_info has category_id but not area idk if I should calculate since the validation does it already
        "segments_info": pred["panoptic_seg"][1]
    }

    pan_img_path = os.path.join(output_dir, img_id + ".png")
    pan_img = pred['panoptic_seg'][0].to("cpu").numpy()

    if test_cfg.use_colors:
        # Using colors breaks the mapping from the color to the segments_info
        # This can be fixed but for now I just generated both greyscale and rgb options

        # Convert the panoptic segmentation to RGB format
        palette = get_color_palette(len(pred["panoptic_seg"][1]))
        pan_img, dict = apply_color_palette(pan_img, palette, dict)

    pan_annotations.append(dict)
    cv2.imwrite(pan_img_path, pan_img)

def print_available_datasets():
    print(DatasetCatalog.keys())

if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    test_cfg = TestConfig()

    cfg = setup_cfg(args)

    if test_cfg.use_extended_categories:
        metadata = extend_open_vocab_dataset()
    else:
        metadata = MetadataCatalog.get("openvocab_ade20k_panoptic_val")

    IDX_TO_CLASS = metadata.stuff_classes
    predictor = DefaultPredictor(cfg)
    if test_cfg.use_oracle:
        # Add dataset mapper to predictor
        mapper = MaskFormerPanopticDatasetMapper(cfg, True, random_flip=False)
        predictor.mapper = mapper
    predictor.set_metadata(metadata)

    os.makedirs(args.output_dir, exist_ok=True)

    pan_annotations = []
    if args.input_dir:
        img_dir = args.input_dir
        img_paths = [os.path.join(img_dir, img_file) for img_file in os.listdir(img_dir)
                      if img_file.endswith((".png", ".jpg"))]
        
        for path in tqdm(img_paths):
            file_exists = os.path.exists(os.path.join(args.output_dir, os.path.basename(path).split(".")[0] + ".png"))
            if test_cfg.skip_seen_files and file_exists:
                continue
            process_image(predictor, path, os.path.basename(path), args.output_dir, pan_annotations, test_cfg)

    elif args.input:
        img_file = args.input.split("/")[-1]
        process_image(predictor, args.input, img_file, args.output_dir, pan_annotations, test_cfg)
    else:
        raise Exception("Input or Input dir required")

    # Construct the output file path
    output_file = os.path.join(args.output_dir, args.annotations_file_name)

    if test_cfg.use_oracle:
        from fcclip.fcclip import MATCHED, OBJECT_COUNT, CATEGORIES_INFO, MISSCLASSIFICATION_INFO, MISSCLASSIFICATION_INFO_BEST_MASKS

        if test_cfg.calculate_confusion_matrix:
            MISSCLASSIFICATION_INFO.print_confusion_matrix()
            MISSCLASSIFICATION_INFO.save_confusion_matrix("./tests/confusion_matrix.txt")
            print('\n')
        
        if test_cfg.calculate_void_clip_classifications:
            MISSCLASSIFICATION_INFO.print_void_clip_metrics()

        if test_cfg.calculate_confusion_matrix_best:
            MISSCLASSIFICATION_INFO_BEST_MASKS.print_confusion_matrix()
            MISSCLASSIFICATION_INFO_BEST_MASKS.save_confusion_matrix("./tests/confusion_matrix_best_masks.txt")

        if test_cfg.calculate_void_clip_classifications_best:
            MISSCLASSIFICATION_INFO_BEST_MASKS.print_void_clip_metrics("./tests/void_clip_classification_best.csv")

        categories_info_ratios = {k: v.miss_count/v.total for k, v in CATEGORIES_INFO.items()}



    with open(output_file, "w") as annotations_file:
        if test_cfg.evaluate:
            json.dump({"annotations": pan_annotations,
                       "missed_objects": 1 - MATCHED/OBJECT_COUNT,
                       "categories_missed_percentage": categories_info_ratios }, annotations_file, indent=4)
        else:
            json.dump({"annotations": pan_annotations}, annotations_file, indent=4)

    # Need to save image_id, file_name and segment_info. image_id seems to be the file_name without extension
    # Look at annotations in validation json for example

    # Segment_info section for sure requires empty area (it is overwritte), category_id. For area maybe set it to 0 to be safe
    # I can also try to calculate it. Area is counts of the unique values

    print("-----------------")