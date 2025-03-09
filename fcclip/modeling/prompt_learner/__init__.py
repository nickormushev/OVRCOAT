import logging
import torch
from detectron2.utils.logger import log_first_n
from fcclip.modeling.prompt_learner.learner import (
    LearnablePromptExtractor,
)

def build_prompt_learner(cfg):
    prompt_learner = LearnablePromptExtractor(
        prompt_dim=cfg.PROMPT.DIM,
        prompt_shape=cfg.PROMPT.SHAPE,
    )
    if cfg.PROMPT.CHECKPOINT != "":
        checkpoint = torch.load(cfg.PROMPT.CHECKPOINT, map_location="cpu")["model"]
        missing, unexpected = prompt_learner.load_state_dict(
            {
                ".".join(k.split(".")[2:]): v
                for k, v in checkpoint.items()
                if "prompt_learner" in k
            },
            strict=False,
        )
        for param in prompt_learner.parameters():
            param.requires_grad = False
        prompt_learner.with_trainable_params = False
        log_first_n(
            logging.INFO,
            "Load Prompt Learner from {}".format(cfg.PROMPT.CHECKPOINT),
            1,
        )
        log_first_n(logging.WARN, "Missing {}".format(missing), 1)
        log_first_n(logging.WARN, "Unexpected {}".format(unexpected), 1)

    else:
        trainable_params = [
            k
            for k, v in prompt_learner.named_parameters()
            if v.requires_grad == True
        ]
        log_first_n(
            logging.INFO,
            "Prompt Learner training params: {}".format(trainable_params),
            1,
        )
    return prompt_learner