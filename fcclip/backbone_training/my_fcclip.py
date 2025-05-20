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

def batched_cosine_similarity_loss(A, B):
    A_normalized = F.normalize(A, p=2, dim=2) 
    B_normalized = F.normalize(B, p=2, dim=2)
    
    cosine_sim = torch.sum(A_normalized * B_normalized, dim=2)  # (B, m)
    
    mean_cosine_sim = torch.mean(cosine_sim, dim=1)  # (B,)
    
    loss = 1 - torch.mean(mean_cosine_sim)  # Scalar loss
    return loss

def linear_warmup(step, total_warmup_steps, final_value):
    if total_warmup_steps == 0:
        return final_value
    return min(final_value, final_value * step / total_warmup_steps)

@META_ARCH_REGISTRY.register()
class MYFCCLIP(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable # calls the configurable wrapper in detectron2.config before init
    def __init__(
        self,
        *,
        backbone: Backbone,
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
        cfg.MODEL.META_ARCHITECTURE = "MYFCCLIP"

        return sem_seg_head, void_embedding
    @classmethod 
    def from_config(cls, cfg): # Called by configurable wrapper before init to get arguments which it passes to init
        # This is the frozen CLIP backbone

        backbone = build_backbone(cfg)
        cfg.defrost()
        cfg.MODEL.BACKBONE.FREEZE = True
        frozen_backbone = build_backbone(cfg)
        cfg.MODEL.BACKBONE.FREEZE = False
        if cfg.TRAIN.USE_PRETRAINED_SEG_HEAD_WEIGHTS:
            sem_seg_head, void_embedding = MYFCCLIP.get_sem_seg_head(cfg)
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
            
        criterion = MYFCCLIP.get_criterion(cfg, sem_seg_head.num_classes)
        MYFCCLIP.cfg = cfg

        return {
            "backbone": backbone,
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

        is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
        cls_prob_no_void = cls_results.softmax(-1)
        mask_cls_probs = torch.cat([
            cls_prob_no_void * (1.0 - is_void_prob),
            is_void_prob], dim=-1)
        mask_cls_results = torch.log(mask_cls_probs + 1e-8)
    
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

        return out_vocab_cls_probs, mask_for_pooling, out_vocab_cls_results
    
    def get_pooled_features(self, clip_features):
        pooled_features = []
        for K in range(1,3):
           pooled_features.append(F.avg_pool2d(clip_features, kernel_size=(K, K)))
        return torch.cat(pooled_features, dim=0)
    
    def calculate_dist_loss(self, clip_feature, frozen_clip_feature):
        # Reshape clip_feature and frozen_clip_feature to [batch_size, num_objects, num_channels * height * width]
        reshaped_clip_feat = clip_feature.view(clip_feature.shape[0], clip_feature.shape[1], -1)
        reshaped_frozen_clip_feat = frozen_clip_feature.view(frozen_clip_feature.shape[0],
                                                            frozen_clip_feature.shape[1], -1)

        # Calculate the Gram matrices using batch matrix multiplication
        gram_matrix_clip_feat = torch.bmm(reshaped_clip_feat, reshaped_clip_feat.transpose(1, 2))
        gram_matrix_frozen_clip_feat = torch.bmm(reshaped_frozen_clip_feat, reshaped_frozen_clip_feat.transpose(1, 2))

        dist_weight = linear_warmup(self.iter, self.dist_warmup_iters, self.weight_dict["dist_loss"])
        self.iter += 1

        if self.loss == "l2":
            return dist_weight * F.mse_loss(gram_matrix_clip_feat, gram_matrix_frozen_clip_feat)
        
        if self.loss == "smoothl1":
            return dist_weight * F.smooth_l1_loss(gram_matrix_clip_feat, gram_matrix_frozen_clip_feat)
        
        if self.loss == "maft":
            pooled_clip_feature = self.get_pooled_features(clip_feature)
            pooled_frozen_clip_feature = self.get_pooled_features(frozen_clip_feature)
            return dist_weight * F.smooth_l1_loss(pooled_clip_feature, pooled_frozen_clip_feature)
        
        return dist_weight * batched_cosine_similarity_loss(gram_matrix_clip_feat, gram_matrix_frozen_clip_feat)
        
    
    def calculate_ce_loss(self, masks, clip_feature, text_classifier, num_templates, gt_labels):
        out_vocab_cls_results, _, _ = self.out_of_vocab_classification(masks, clip_feature, text_classifier, num_templates)
        batch_size, num_masks, num_classes = out_vocab_cls_results.shape

        # Reshape out_vocab_cls_results to [batch_size * num_objects, num_classes]
        out_vocab_cls_results = out_vocab_cls_results.reshape(batch_size * num_masks, num_classes)

        # Reshape gt_labels to [batch_size * num_objects]
        gt_labels = gt_labels.reshape(batch_size * num_masks)
        return F.cross_entropy(out_vocab_cls_results, gt_labels, reduction='mean')

    def train_with_generated_masks(self, targets, seg_head_features, clip_feature, text_classifier, num_templates, frozen_clip_feature):
        outputs = self.sem_seg_head(seg_head_features)
        pred_masks = outputs["pred_masks"]
        if self.detach_seg_head:
            pred_masks = pred_masks.detach()

        oov_cls_res, _, _ = self.out_of_vocab_classification(pred_masks, clip_feature, text_classifier, num_templates)
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

        dist_loss = self.calculate_dist_loss(clip_feature, frozen_clip_feature)
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
            ce_loss += self.calculate_ce_loss(gt_masks, clip_feature[i:i+1], text_classifier, num_templates, gt_labels)

        ce_loss = ce_loss / len(targets)
        dist_loss = self.calculate_dist_loss(clip_feature, frozen_clip_feature)

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

    def test_with_imperfect_masks(self, clip_feature, frozen_features,
                                 features, text_classifier, num_templates, images):

        seg_head_features = self.get_seg_head_features(features, frozen_features,
                                                text_classifier, num_templates)
        mask_2_former_outputs = self.sem_seg_head(seg_head_features)

        mask_pred_results = mask_2_former_outputs["pred_masks"]
        oov_cls_probs, mask_for_pooling, similarities = self.out_of_vocab_classification(mask_pred_results,
                                        clip_feature, text_classifier, num_templates)

        self.clip_preds = oov_cls_probs
        assert not (self.test_with_fc_clip and self.test_with_void), "You cannot use void and fc-clip at the same time"
        if self.test_with_fc_clip:
            mask_cls_results = self.ensemble_classifications(mask_2_former_outputs["pred_logits"],
                                                    oov_cls_probs, mask_for_pooling)
        else:
            mask_cls_results = oov_cls_probs

            # TODO: Set this from config only for evaluation of some models
            # Or retrain those models
            if self.test_with_void:
                clip_res = mask_2_former_outputs["pred_logits"]
                is_void_prob = F.softmax(clip_res, dim=-1)[..., -1:]
                mask_cls_probs = torch.cat([
                    mask_cls_results * (1.0 - is_void_prob),
                    is_void_prob], dim=-1)
                mask_cls_results = torch.log(mask_cls_probs + 1e-8)

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        return mask_cls_results, mask_pred_results, similarities, oov_cls_probs
    
    def reclassify_void_masks(self, num_classes, pred_clfs, clip_preds, clip_similarities,
                                give_things_pirority = True, clip_treshold=0.8,
                                sim_threshold=26, softmax_temperature=6):

        pred_clfs_np = pred_clfs.cpu().detach().numpy()
        clip_preds_np = clip_preds.cpu().detach().numpy()
        new_mask_cls = pred_clfs_np

        for i in range(pred_clfs_np.shape[0]): 
            pred_category = np.argmax(pred_clfs_np[i])
            pred_is_background = pred_category == (num_classes - 1)
            clip_category = np.argmax(clip_preds_np[i])
            clip_prob = np.max(clip_preds_np[i])
            similarity = clip_similarities[i, clip_category]

            is_thing = clip_category in self.test_metadata.thing_dataset_id_to_contiguous_id.values()

            if pred_is_background and clip_prob >= clip_treshold and similarity > sim_threshold:
                new_mask_cls[i, 0:num_classes - 1] = F.softmax(clip_similarities[i]/softmax_temperature, dim=-1).cpu().detach().numpy()
                new_mask_cls[i, num_classes - 1] = 0

                if is_thing and give_things_pirority:
                    new_mask_cls[i, clip_category] += 0.1 # Gives priority to things.

        return torch.tensor(new_mask_cls, device=self.device)

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

       # if not self.init_embedding:
       #    model = build_model(self.cfg)
       #    checkpointer = DetectionCheckpointer(model)
       #    checkpointer.load(self.cfg.MODEL.FC_CLIP.SEM_SEG_WEIGHTS)
       #    self.void_embedding = model.void_embedding
       #    self.init_embedding = True

        if not self.train_seg_head:
            self.sem_seg_head.eval()

        features = self.backbone(images.tensor)
        with torch.no_grad():
            frozen_features = self.frozen_backbone(images.tensor)
        
        clip_feature = features["clip_vis_dense"] # Last layer/output of features of ConvNeXt/CLIP
        frozen_clip_feature = frozen_features["clip_vis_dense"]

        text_classifier, num_templates = self.get_text_classifier()
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)

        if self.training:
            targets = self.get_targets(batched_inputs, images)

            if self.train_with_fc_clip_masks:
                seg_head_features = self.get_seg_head_features(features, frozen_features, text_classifier, num_templates)
                return self.train_with_generated_masks(targets, seg_head_features, clip_feature,
                                                    text_classifier, num_templates, frozen_clip_feature)
            else:
                return self.train_with_gt_masks(targets, clip_feature, text_classifier,
                                            num_templates, frozen_clip_feature)
        else:
            with torch.no_grad():
                if self.test_perfect_masks:
                    targets = self.get_targets(batched_inputs, images)
                    mask_pred_results = targets[0]['masks'].unsqueeze(0).float()
            
                    out_vocab_cls_probs, _, _ = self.out_of_vocab_classification(mask_pred_results,
                                                                clip_feature, text_classifier, num_templates)

                    mask_pred_results = mask_pred_results.to(self.device)
                    mask_cls_results = out_vocab_cls_probs
                else:
                    mask_cls_results, mask_pred_results, similarities, out_vocab_cls_probs = self.test_with_imperfect_masks(clip_feature,
                                            frozen_features, features, text_classifier,
                                            num_templates, images)

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)( # Is literally just upsampling
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    if self.test_with_void or self.test_with_fc_clip:
                        mask_cls_result = F.softmax(mask_cls_result, dim=-1)

                        if self.reclassify_void:
                            num_classes = mask_cls_result.shape[1]
                            mask_cls_result = self.reclassify_void_masks(num_classes, mask_cls_result, out_vocab_cls_probs[0], similarities[0])
                    
                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result) # Multiplies class with mask results
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results

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

    def semantic_inference(self, mask_cls, mask_pred):
        if not self.test_perfect_masks:
            if self.test_with_void or self.test_with_fc_clip:
                mask_cls = mask_cls[..., :-1]
            mask_pred = mask_pred.sigmoid()

        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = mask_cls.max(-1) # For each pixel, get the class with the highest score

        if not self.test_perfect_masks:
            mask_pred = mask_pred.sigmoid()
            #mask_pred = mask_pred > 0.5 # Binarize the masks

        num_classes = len(self.test_metadata.stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold) # Thresholding I guess. First part removes background I think
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks # Each pixel in the mask has the value of the score. Think that masks are 0,1

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0) # Gets the max score for the pixel. Actually the max index. Uses argmax
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5) # Takes pixels of mask but only ones we are sure of 

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold: # The mask is covered by another
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = mask_cls.to(self.device)

        if self.test_with_void or self.test_with_fc_clip:
            scores = scores[:, :-1]

        # if this is panoptic segmentation
        if self.panoptic_on:
            num_classes = len(self.test_metadata.stuff_classes)
        else:
            num_classes = len(self.test_metadata.thing_classes)
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(mask_pred.shape[0], 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        topk_indices.to(self.device)
        scores_per_image.to(self.device)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices].to(self.device)

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(labels_per_image).bool().to(self.device)
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.test_metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
