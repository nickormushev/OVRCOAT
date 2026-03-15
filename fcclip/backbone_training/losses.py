import torch
from torch.nn import functional as F


def calculate_dist_loss(clip_feature, frozen_clip_feature, weight_dict):
    # Reshape clip_feature and frozen_clip_feature to [batch_size, num_channels * height * width]
    reshaped_clip_feat = clip_feature.view(
        clip_feature.shape[0], clip_feature.shape[1], -1
    )
    reshaped_frozen_clip_feat = frozen_clip_feature.view(
        frozen_clip_feature.shape[0], frozen_clip_feature.shape[1], -1
    )

    reshaped_clip_feat = F.normalize(
        reshaped_clip_feat, dim=2
    )  # normalise along spatial dims
    reshaped_frozen_clip_feat = F.normalize(reshaped_frozen_clip_feat, dim=2)

    gram_matrix_clip_feat = torch.bmm(
        reshaped_clip_feat, reshaped_clip_feat.transpose(1, 2)
    )
    gram_matrix_frozen_clip_feat = torch.bmm(
        reshaped_frozen_clip_feat, reshaped_frozen_clip_feat.transpose(1, 2)
    )

    return weight_dict["dist_loss"] * F.mse_loss(
        gram_matrix_clip_feat, gram_matrix_frozen_clip_feat
    )
