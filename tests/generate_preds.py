import torch
import sys
import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm 
import itertools

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

from detectron2.engine.defaults import DefaultPredictor as d2_defaultPredictor

class DefaultPredictor(d2_defaultPredictor):
    def set_metadata(self, metadata):
        self.model.set_metadata(metadata)

# TODO: Research if there is a better way to do this cause this is a bit insane
new_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(new_root_dir)
sys.path.append(new_root_dir)

from fcclip.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from fcclip import add_maskformer2_config, add_fcclip_config
from detectron2.modeling import build_model
 
# BatchPredictor is not used/implemented for now due to reasonable inference time with DefaultPredictor
#class BatchPredictor:
#    """
#    Create a simple end-to-end predictor with the given config that runs on
#    single device for a single input image.
#
#    Compared to using the model directly, this class does the following additions:
#
#    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
#    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
#    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
#    4. Take one input image and produce a single output, instead of a batch.
#
#    This is meant for simple demo purposes, so it does the above steps automatically.
#    This is not meant for benchmarks or running complicated inference logic.
#    If you'd like to do anything more complicated, please refer to its source code as
#    examples to build and use the model manually.
#
#    Attributes:
#        metadata (Metadata): the metadata of the underlying dataset, obtained from
#            cfg.DATASETS.TEST.
#
#    Examples:
#    ::
#        pred = DefaultPredictor(cfg)
#        inputs = cv2.imread("input.jpg")
#        outputs = pred(inputs)
#    """
#
#    def __init__(self, cfg):
#        self.cfg = cfg.clone()  # cfg can be modified by model
#        self.model = build_model(self.cfg)
#        self.model.eval()
#        if len(cfg.DATASETS.TEST):
#            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
#
#        checkpointer = DetectionCheckpointer(self.model)
#        checkpointer.load(cfg.MODEL.WEIGHTS)
#
#        self.aug = T.ResizeShortestEdge(
#            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
#        )
#
#        self.input_format = cfg.INPUT.FORMAT
#        assert self.input_format in ["RGB", "BGR"], self.input_format
#
#    def __call__(self, original_image):
#        """
#        Args:
#            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
#
#        Returns:
#            predictions (dict):
#                the output of the model for one image only.
#                See :doc:`/tutorials/models` for details about the format.
#        """
#        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
#            # Apply pre-processing to image.
#            if self.input_format == "RGB":
#                # whether the model expects BGR inputs or RGB
#                original_image = original_image[:, :, ::-1]
#            height, width = original_image.shape[:2]
#            image = self.aug.get_transform(original_image).apply_image(original_image)
#            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
#            image.to(self.cfg.MODEL.DEVICE)
#
#            inputs = {"image": image, "height": height, "width": width}
#
#            predictions = self.model([inputs])
#            return predictions

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
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
        default="./tests/preds",
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
    return np.random.randint(0, 255, size=(num_colors, 3), dtype=np.uint8)

def apply_color_palette(segmentation, palette):
    h, w = segmentation.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(segmentation)
    for uid in unique_ids:
        colored_mask[segmentation == uid] = palette[uid % len(palette)]
    return colored_mask

def process_image(predictor, img_path, img_file, output_dir, pan_annotations):
    img = read_image(img_path, format="BGR")
    pred = predictor(img)
    img_id = img_file.split(".")[0]
    dict = {
        "image_id": img_id,
        "file_name": img_id + ".png",
        # Segment_info has category_id but not area idk if I should calculate since the validation does it already
        "segments_info": pred["panoptic_seg"][1]
    }

    pan_annotations.append(dict)

    # cv2.imwrite(pan_img_path, pred['panoptic_seg'][0].to("cpu").numpy())
    # Convert the panoptic segmentation to RGB format
    pan_img = pred['panoptic_seg'][0].to("cpu").numpy()
    palette = get_color_palette(len(pred["panoptic_seg"][1]))
    pan_img_rgb = apply_color_palette(pan_img, palette)

    # Save the panoptic segmentation image in RGB format
    pan_img_path = os.path.join(output_dir, img_id + ".png")
    cv2.imwrite(pan_img_path, pan_img_rgb)

def print_available_datasets():
    print(DatasetCatalog.keys())

def worker(proc_id, predictor, img_paths, output_dir):
    local_annotations = []
    for img_path in img_paths:
        img_file = os.path.basename(img_path)
        process_image(predictor, img_path, img_file, output_dir, local_annotations)
    
    return local_annotations

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    #dataset = DatasetCatalog.get("openvocab_ade20k_panoptic_train")
    #metadata = MetadataCatalog.get("openvocab_ade20k_panoptic_val")
    metadata = create_open_vocab_dataset()

    predictor = DefaultPredictor(cfg)
    predictor.set_metadata(metadata)

    os.makedirs(args.output_dir, exist_ok=True)

    pan_annotations = []
    if args.input_dir:
        img_dir = args.input_dir
        img_paths = [os.path.join(img_dir, img_file) for img_file in os.listdir(img_dir)
                      if img_file.endswith((".png", ".jpg"))]
        
        cpu_num = 4
        chunk_size = len(img_paths) // cpu_num
        chunks = [img_paths[i:i + chunk_size] for i in range(0, len(img_paths), chunk_size)]

        workers = mp.Pool(processes=cpu_num)
        processes = []
        for proc_id, chunk in enumerate(chunks):
            p = workers.apply_async(worker, (proc_id, predictor, chunk, args.output_dir))
            processes.append(p)
        

        workers.close()
        workers.join()

        for p in processes:
            s = p.get()
            pan_annotations.extend(s)

    elif args.input:
        img_file = args.input.split("/")[-1]
        process_image(predictor, args.input, img_file, args.output_dir, pan_annotations)
    else:
        raise Exception("Input or Input dir required")


    # Construct the output file path
    output_file = os.path.join(args.output_dir, args.annotations_file_name)


    # Write annotations to the output file
    with open(output_file, "w") as annotations_file:
        json.dump({"annotations": pan_annotations}, annotations_file, indent=4)

    # Need to save image_id, file_name and segment_info. image_id seems to be the file_name without extension
    # Look at annotations in validation json for example

    # Segment_info section for sure requires empty area (it is overwritte), category_id. For area maybe set it to 0 to be safe
    # I can also try to calculate it. Area is counts of the unique values

    print("-----------------")