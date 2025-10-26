import wandb
"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
"""
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head, build_model
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.checkpoint import DetectionCheckpointer

from fcclip.modeling.criterion import SetCriterion
from fcclip.modeling.matcher import HungarianMatcher

from fcclip.backbone_training.mask_aware_loss import MA_Loss
from fcclip.backbone_training.inference_utils import panoptic_inference, semantic_inference, instance_inference
from fcclip.backbone_training.losses import calculate_dist_loss, calculate_ce_loss

from fcclip.modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits
VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]

MATCHED = 0
OBJECT_COUNT = 0

# ADE20K specific. Is number of classes - 1 otherwise
VOID_CATEGORY_ID = 150

@META_ARCH_REGISTRY.register()
class RECLIP(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable # calls the configurable wrapper in detectron2.config before init
    def __init__(
        self,
        *,
        backbone: Backbone,
        use_dist_loss: bool,
        frozen_backbone: Backbone,
        sem_seg_head: nn.Module,
        void_embedding: nn.Embedding,
        criterion: nn.Module,
        weight_dict: dict,
        train_with_fc_clip_masks: bool,
        use_tuned_features_for_seg_head: bool,
        train_seg_head: bool,
        detach_seg_head: bool,
        test_perfect_masks: bool,
        use_ma_loss: bool,
        test_with_void: bool,
        test_with_fc_clip: bool,
        dist_warmup_iters: int,
        loss: str,
        use_pooling_weights: bool,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        train_metadata,
        test_metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        reclassify_void: bool,
        # FC-CLIP
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        ensemble_on_valid_mask: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.weight_dict = weight_dict
        self.frozen_backbone = frozen_backbone
        self.dist_warmup_iters = dist_warmup_iters
        self.test_perfect_masks = test_perfect_masks
        self.test_with_void = test_with_void
        self.test_with_fc_clip = test_with_fc_clip
        self.iter = 0
        self.sem_seg_head = sem_seg_head
        if not train_seg_head:
            self.sem_seg_head.eval()
        self.loss = loss
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)


        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.reclassify_void = reclassify_void

        self.train_with_fc_clip_masks = train_with_fc_clip_masks
        self.train_seg_head = train_seg_head
        self.detach_seg_head = detach_seg_head
        self.use_tuned_features_for_seg_head = use_tuned_features_for_seg_head
        self.criterion = criterion

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # FC-CLIP args
        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.ensemble_on_valid_mask = ensemble_on_valid_mask 

        self.use_ma_loss = use_ma_loss
        self.ma_loss = MA_Loss()  # BCELoss BCEWithLogitsLoss SmoothL1Loss
        self.use_dist_loss = use_dist_loss

        self.train_text_classifier = None
        self.test_text_classifier = None

        self.void_embedding = void_embedding
        self.init_embedding = False

        self.use_pooling_weights = use_pooling_weights

        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(test_metadata, train_metadata)

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)
        
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)
       
        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        return category_overlapping_mask, num_templates, class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return

    def get_text_classifier(self):
        if self.training:
            if self.train_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.train_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    # For each class generates embeddings for each template with the text encoder
                    text_classifier.append(self.backbone.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
                # The generated  embedings are concatenated
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            # First is the embeddings for the prompts with each class 
            # and the second is the number of templates per class used
            return self.test_text_classifier, self.test_num_templates

    @classmethod
    def get_criterion(cls, cfg, num_classes):
        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        oov_weight = cfg.MODEL.FC_CLIP.CE_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            cost_oov=oov_weight,
        )

        weight_dict = {"loss_ce": class_weight,
                        "loss_mask": mask_weight,
                        "loss_dice": dice_weight,
                        "loss_oov_ce": oov_weight}

        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        #losses = ["labels", "masks", "oov_ce"]
        losses = cfg.TRAIN.LOSSES
        return SetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

    @classmethod
    def get_sem_seg_head(cls, cfg):
        cfg.MODEL.META_ARCHITECTURE = "FCCLIP"
        cls.cfg = cfg
        model = build_model(cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.FC_CLIP.SEM_SEG_WEIGHTS)
        sem_seg_head = model.sem_seg_head
        void_embedding = model.void_embedding

        if not cfg.TRAIN.SEG_HEAD:
            # Freeze all parameters of sem_seg_head
            for param in sem_seg_head.parameters():
                param.requires_grad = False
            void_embedding.requires_grad = False

        del model
        cfg.MODEL.META_ARCHITECTURE = "RECLIP"

        return sem_seg_head, void_embedding

    @classmethod 
    def from_config(cls, cfg): # Called by configurable wrapper before init to get arguments which it passes to init
        # This is the frozen CLIP backbone

        backbone = build_backbone(cfg)
        cfg.defrost()
        old_freeze = cfg.MODEL.BACKBONE.FREEZE
        cfg.MODEL.BACKBONE.FREEZE = True
        frozen_backbone = build_backbone(cfg)
        cfg.MODEL.BACKBONE.FREEZE = old_freeze

        if cfg.TRAIN.USE_PRETRAINED_SEG_HEAD_WEIGHTS:
            sem_seg_head, void_embedding = RECLIP.get_sem_seg_head(cfg)
        else:
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
            void_embedding = nn.Embedding(1, backbone.dim_latent)
        cfg.freeze()

        dist_weight = cfg.MODEL.FC_CLIP.DIST_WEIGHT
        ce_weight = cfg.MODEL.FC_CLIP.CE_WEIGHT

        weight_dict = {
            "ce_loss": ce_weight,
            "dist_loss": dist_weight,
        }

        # TODO: Make this cleaner
        use_ma_loss = "ma_loss" in cfg.TRAIN.LOSSES
        if use_ma_loss:
            cfg.TRAIN.LOSSES.remove("ma_loss")
            
        criterion = RECLIP.get_criterion(cfg, sem_seg_head.num_classes)
        RECLIP.cfg = cfg

        return {
            "backbone": backbone,
            "use_dist_loss": not cfg.MODEL.BACKBONE.FREEZE, # If tuning backbone, use distillation loss
            "weight_dict": weight_dict,
            "reclassify_void": cfg.MODEL.RECLASSIFY_VOID,
            "frozen_backbone": frozen_backbone,
            "void_embedding": void_embedding,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "train_with_fc_clip_masks": cfg.TRAIN.WITH_FC_CLIP_MASKS,
            "train_seg_head": cfg.TRAIN.SEG_HEAD,
            "use_tuned_features_for_seg_head": cfg.TRAIN.USE_TUNED_FEATURES_FOR_SEG_HEAD,
            "use_ma_loss": use_ma_loss,
            "detach_seg_head": cfg.TRAIN.DETACH_SEG_HEAD,
            "test_with_void": cfg.TEST.WITH_VOID,
            "test_with_fc_clip": cfg.TEST.WITH_FC_CLIP,
            "test_perfect_masks": cfg.TEST.PERFECT_MASKS,
            "dist_warmup_iters": cfg.MODEL.FC_CLIP.DIST_WARMUP_ITERS,
            "loss": cfg.MODEL.FC_CLIP.LOSS,
            "use_pooling_weights": cfg.MODEL.FC_CLIP.USE_POOLING_WEIGHTS,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "geometric_ensemble_alpha": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def ensemble_classifications(self, mask_cls_results, out_vocab_cls_probs, mask_for_pooling):
        in_vocab_cls_results = mask_cls_results[..., :-1] # remove void

        in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
        category_overlapping_mask = self.category_overlapping_mask.to(self.device) # Says if a pixel is seen before

        if self.ensemble_on_valid_mask:
            # Only include out_vocab cls results on masks with valid pixels
            # We empirically find that this is important to obtain reasonable AP/mIOU score with ResNet CLIP models
            valid_masking = (mask_for_pooling > 0).to(mask_for_pooling).sum(-1).sum(-1) > 0
            valid_masking = valid_masking.to(in_vocab_cls_results.dtype).unsqueeze(-1)
            alpha = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_alpha
            beta = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_beta
            alpha = alpha * valid_masking
            beta = beta * valid_masking
        else:
            alpha = self.geometric_ensemble_alpha
            beta = self.geometric_ensemble_beta

        cls_logits_seen = (
            (in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs**alpha).log()
            * category_overlapping_mask # If pixel is seen during training we use this classifier
        )
        cls_logits_unseen = (
            (in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs**beta).log()
            * (1 - category_overlapping_mask) # If pixel not seen during training we use this classifier
        )

        cls_results = cls_logits_seen + cls_logits_unseen # Combine predictions
        cls_prob_no_void = cls_results.softmax(-1)

        mask_cls_results = self.add_void_probability(cls_prob_no_void, mask_cls_results, out_vocab_cls_probs)
    
        return mask_cls_results

    def out_of_vocab_classification(self, masks, clip_features, text_classifier, num_templates, interpolate_mode="bilinear"):
        mask_for_pooling = F.interpolate(masks, size=clip_features.shape[-2:], mode=interpolate_mode, align_corners=False)

        if "convnext" in self.backbone.model_name.lower():
            pooled_clip_feature = self.mask_pooling(clip_features, mask_for_pooling)  # Apply pooling with mask and get embedding
            pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
        elif "rn" in self.backbone.model_name.lower():
            pooled_clip_feature = self.backbone.visual_prediction_forward(clip_features, mask_for_pooling)
        else:
            raise NotImplementedError

        out_vocab_cls_results = get_classification_logits(pooled_clip_feature, text_classifier,
                                                        self.backbone.clip_model.logit_scale, num_templates)
        out_vocab_cls_results = out_vocab_cls_results[..., :-1]
        out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)

        return out_vocab_cls_probs, mask_for_pooling

    def train_with_generated_masks(self, targets, seg_head_features, clip_feature, text_classifier, num_templates, frozen_clip_feature):
        outputs = self.sem_seg_head(seg_head_features)
        pred_masks = outputs["pred_masks"]
        if self.detach_seg_head:
            pred_masks = pred_masks.detach()

        oov_cls_res, _ = self.out_of_vocab_classification(pred_masks, clip_feature, text_classifier, num_templates)
        outputs["oov_cls_res"] = oov_cls_res

        # FC-CLIP criterion extended with oov_ce loss
        losses = self.criterion(outputs, targets) if self.criterion.losses else None

        if self.train_seg_head and losses:
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
        
        if self.use_ma_loss:
            ranking_loss = self.ma_loss(oov_cls_res, pred_masks, targets)
            losses = {"ranking_loss": ranking_loss} if losses is None else \
                     {**losses, "ranking_loss": ranking_loss}

        if self.use_dist_loss:
            dist_loss = calculate_dist_loss(clip_feature, frozen_clip_feature, self.loss, self.iter, self.dist_warmup_iters, self.weight_dict)
            losses["dist_loss"] = dist_loss

        wandb.log(losses)
        return losses

    def train_with_gt_masks(self, targets, clip_feature, text_classifier, num_templates, frozen_clip_feature):
        ce_loss = 0
        for i, targets_per_img in enumerate(targets):
            gt_masks = targets_per_img["masks"]
            gt_labels = targets_per_img["labels"]

            num_masks = gt_masks.shape[0]
            if num_masks == 0:
                continue

            gt_masks = gt_masks.unsqueeze(0).float()
            out_vocab_cls_results, _ = self.out_of_vocab_classification(gt_masks, clip_feature[i:i+1], text_classifier, num_templates)
            ce_loss += calculate_ce_loss(out_vocab_cls_results, gt_labels)

        ce_loss = ce_loss / len(targets)
        dist_loss = calculate_dist_loss(clip_feature, frozen_clip_feature, self.loss, self.iter, self.dist_warmup_iters, self.weight_dict)

        losses = {
            "oov_ce_loss": ce_loss * self.weight_dict["ce_loss"],
            "dist_loss": dist_loss,
        }
        wandb.log(losses)
        return losses
    
    def get_seg_head_features(self, features, frozen_features, text_classifier, num_templates):
        if self.train_seg_head and self.use_tuned_features_for_seg_head:
            seg_head_features = features.copy()

            if self.detach_seg_head:
                for k in seg_head_features.keys():
                    seg_head_features[k] = seg_head_features[k].detach()
        else:
            seg_head_features = frozen_features

        seg_head_features['text_classifier'] = text_classifier
        seg_head_features['num_templates'] = num_templates

        return seg_head_features

    def get_targets(self, batched_inputs, images):
        # mask classification target
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            return self.prepare_targets(gt_instances, images)

        return None
    
    def add_void_probability(self, cls_results, mask2former_cls_res, out_vocab_cls_probs):
        is_void_prob = F.softmax(mask2former_cls_res, dim=-1)[..., -1:]
        max_clip_prob, _ = out_vocab_cls_probs.max(-1, keepdim=True)
        weight = 0.5
        is_void_prob = is_void_prob * (1 - weight * max_clip_prob)
        mask_cls_probs = torch.cat([
            cls_results * (1.0 - is_void_prob),
            is_void_prob], dim=-1)
        return torch.log(mask_cls_probs + 1e-8)
    
    def get_mask_cls_results(self, mask_2_former_outputs, oov_cls_probs, mask_for_pooling):
        assert not (self.test_with_fc_clip and self.test_with_void), "You cannot use void and fc-clip at the same time"
        if self.test_with_fc_clip:
            return self.ensemble_classifications(mask_2_former_outputs["pred_logits"],
                                            oov_cls_probs, mask_for_pooling)
        elif self.test_with_void:
            return self.add_void_probability(oov_cls_probs,
                                        mask_2_former_outputs["pred_logits"])
        return oov_cls_probs

    def test_with_imperfect_masks(self, clip_feature, frozen_features,
                                 features, text_classifier, num_templates, images):

        seg_head_features = self.get_seg_head_features(features, frozen_features,
                                                text_classifier, num_templates)
        mask_2_former_outputs = self.sem_seg_head(seg_head_features)

        mask_pred_results = mask_2_former_outputs["pred_masks"]
        oov_cls_probs, mask_for_pooling = self.out_of_vocab_classification(mask_pred_results,
                                        clip_feature, text_classifier, num_templates)

        mask_cls_results = self.get_mask_cls_results(mask_2_former_outputs, oov_cls_probs, mask_for_pooling)

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        return mask_cls_results, mask_pred_results
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # ImageList stores images in varying shapes by padding them to same size
        images = ImageList.from_tensors(images, self.size_divisibility)

        #if not self.init_embedding:
        #   model = build_model(self.cfg)
        #   checkpointer = DetectionCheckpointer(model)
        #   checkpointer.load(self.cfg.MODEL.FC_CLIP.SEM_SEG_WEIGHTS)
        #   self.void_embedding = model.void_embedding
        #   self.init_embedding = True

        if not self.train_seg_head:
            self.sem_seg_head.eval()

        features = self.backbone(images.tensor)
        with torch.no_grad():
            frozen_features = self.frozen_backbone(images.tensor)
        
        text_classifier, num_templates = self.get_text_classifier()
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)

        if self.training:
            return self.train_step(batched_inputs, images, features, frozen_features, text_classifier, num_templates)

        return self.inference_step(batched_inputs, images, features, frozen_features, text_classifier, num_templates)
        
    def train_step(self, batched_inputs, images, features, frozen_features, text_classifier, num_templates):
        targets = self.get_targets(batched_inputs, images)
        clip_feature = features["clip_vis_dense"] # Last layer/output of features of ConvNeXt/CLIP
        frozen_clip_feature = frozen_features["clip_vis_dense"]

        if self.train_with_fc_clip_masks:
            seg_head_features = self.get_seg_head_features(features, frozen_features, text_classifier, num_templates)
            return self.train_with_generated_masks(targets, seg_head_features, clip_feature,
                                                text_classifier, num_templates, frozen_clip_feature)

        return self.train_with_gt_masks(targets, clip_feature, text_classifier,
                                    num_templates, frozen_clip_feature)
    
    def inference_step(self, batched_inputs, images, features, frozen_features,
                       text_classifier, num_templates):

        clip_feature = features["clip_vis_dense"] # Last layer/output of features of ConvNeXt/CLIP
        with torch.no_grad():
            if self.test_perfect_masks:
                targets = self.get_targets(batched_inputs, images)
                mask_pred_results = targets[0]['masks'].unsqueeze(0).float()
            
                out_vocab_cls_probs, _ = self.out_of_vocab_classification(mask_pred_results,
                                                            clip_feature, text_classifier, num_templates)

                mask_pred_results = mask_pred_results.to(self.device)
                mask_cls_results = out_vocab_cls_probs
            else:
                mask_cls_results, mask_pred_results = self.test_with_imperfect_masks(clip_feature,
                                        frozen_features, features, text_classifier,
                                        num_templates, images)

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                torch.cuda.empty_cache()
                result = self.process_single_image(mask_cls_result, mask_pred_result, input_per_image, \
                    image_size)
                processed_results.append(result)

            return processed_results

    def process_single_image(self, mask_cls_result, mask_pred_result, input_per_image, image_size):
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])

        if self.sem_seg_postprocess_before_inference:
            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
            mask_cls_result = mask_cls_result.to(mask_pred_result)

        if self.test_with_void or self.test_with_fc_clip:
            mask_cls_result = F.softmax(mask_cls_result, dim=-1)
            
        return self.perform_inference(mask_cls_result, mask_pred_result, image_size, height, width)

    
    def perform_inference(self, mask_cls_result, mask_pred_result, image_size, height, width):
        result = {}
        # Semantic segmentation inference
        if self.semantic_on:
            r = retry_if_cuda_oom(semantic_inference)(mask_cls_result, mask_pred_result,
                                        self.test_perfect_masks, self.test_with_void, self.test_with_fc_clip)
            if not self.sem_seg_postprocess_before_inference:
                r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
            result["sem_seg"] = r

        # Panoptic segmentation inference
        if self.panoptic_on:
            panoptic_r = retry_if_cuda_oom(panoptic_inference)(mask_cls_result, mask_pred_result, self.test_perfect_masks,
                       self.reclassify_void, self.test_metadata, self.overlap_threshold, self.object_mask_threshold)
            result["panoptic_seg"] = panoptic_r

        # Instance segmentation inference
        if self.instance_on:
            instance_r = retry_if_cuda_oom(instance_inference)(mask_cls_result, mask_pred_result, self.test_topk_per_image,
                        self.test_with_void, self.test_with_fc_clip, self.panoptic_on, self.test_metadata, self.device)
            result["instances"] = instance_r

        return result

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets