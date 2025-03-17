import torch
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config
from fcclip import add_maskformer2_config, add_fcclip_config
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head, build_model
import argparse
import os

# Small script to convert old models to new format with more parameters.


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.MODEL.BACKBONE.FREEZE = True
    cfg.MODEL.BACKBONE.PROMPT = CN()
    cfg.MODEL.BACKBONE.PROMPT.LEARNABLE = False
    cfg.MODEL.BACKBONE.PROMPT.DIM = 512
    cfg.MODEL.BACKBONE.PROMPT.SHAPE = (16, 0)
    cfg.MODEL.BACKBONE.PROMPT.CHECKPOINT = ""

    cfg.TEST.PERFECT_MASKS = False
    cfg.TEST.WITH_VOID = False
    cfg.TEST.WITH_FC_CLIP = False

    # TRAIN options are not used with generate preds but required to build model
    cfg.TRAIN = CN()
    cfg.TRAIN.SEG_HEAD = False
    cfg.TRAIN.WITH_FC_CLIP_MASKS = False
    cfg.TRAIN.LOSSES = ["masks", "labels", "oov_ce"]
    cfg.TRAIN.USE_TUNED_FEATURES_FOR_SEG_HEAD = False
    cfg.TRAIN.DETACH_SEG_HEAD = False
    cfg.TRAIN.USE_PRETRAINED_SEG_HEAD_WEIGHTS = True

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
        default="configs/coco/panoptic-segmentation/fcclip/my_fcclip_convnext_large_eval_coco_r50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="modify config options using the command-line 'key value' pairs",
        # hardcoded default
        default=["MODEL.WEIGHTS", "/home/nikolay/Downloads/fcclip_cocopan.pth"],
        nargs=argparse.REMAINDER,
    )

    return parser

# A small check to see if the weights have changed after training.
# For debugging purposes
def check_change_in_params(state_dict):
    sem_seg_head_orig = torch.load(f"/home/nikolay/fcclip_cocopan_r50.pth")
    sem_seg_head_state_dict = sem_seg_head_orig["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in sem_seg_head_state_dict:
        if key in state_dict:
            sem_seg_head_state_dict[key] = sem_seg_head_state_dict[key].to(device)
            state_dict[key] = state_dict[key].to(device)
            if not torch.equal(sem_seg_head_state_dict[key], state_dict[key]):
                print(f"Difference found in weight: {key}")

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    model_path = "TRAIN_L2_001"
    iters = ["0019999"]
    for iter in iters:
        checkpoint = torch.load(f"./{model_path}/model_{iter}.pth")
        state_dict = checkpoint["model"]

        model = build_model(cfg)
        model.load_state_dict(state_dict, strict=False)
        os.makedirs(f"./{model_path}_UPDATED", exist_ok=True)
        torch.save(model.state_dict(), f"./{model_path}_UPDATED/model_{iter}.pth")


