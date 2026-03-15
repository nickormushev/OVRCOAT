from . import (
    data,  # register all new datasets
    modeling,
)

# config
from .config import add_fcclip_config, add_maskformer2_config, add_ovrcoat_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
    COCOInstanceNewBaselineDatasetMapper,
)
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import (
    COCOPanopticNewBaselineDatasetMapper,
)
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.panoptic_evaluation import COCOPanopticEvaluator

# models
from .fcclip import FCCLIP
from .ovrcoat import OVRCOAT
from .test_time_augmentation import SemanticSegmentorWithTTA
