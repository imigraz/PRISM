"""U-Net used by PRISM: the nnU-Net 2D U-Net with input sparsification.

Invariant sparse masking (ISM) applies channel-consistent dropout (CCD): one Bernoulli mask per spatial
position, shared by all channels and rescaled by 1 / (1 - rate), followed by local normalization of the
remaining entries (Uhrig et al., 2017) when at least 90% of the masked feature map is zero. It is used in two
places, each with its own rate:

* ``sparsity_aug``: as augmentation, on a quarter of the current-domain batch at every encoder level
  during training (``sparsify_features(x, during_normal_training=True)``);
* ``sparsity_replay``: on the feature maps extracted for the replay buffer
  (``forward(x, layer_name_for_feature_extraction=...)``, called by the sliding-window feature extraction).

ISM is not applied at inference. Trainers create a ``Generic_UNet_`` and switch its ``__class__`` to this
class, so ``__init__`` does not run there; they set the two rates with ``set_sparsity`` after each switch.
"""
from typing import List

import torch
import torch.nn.functional as F
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn

from nnunet_ext.network_architecture.GenericUNet_ import Generic_UNet_


class Generic_UNet_PRISM(Generic_UNet_):
    """Generic_UNet_ with input sparsification, encoder freezing and replay from first-level features."""

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size=None, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False,
                 no_skips=False):
        # patch_size is not used; it only keeps the positional signature.
        super(Generic_UNet_PRISM, self).__init__(input_channels, base_num_features, num_classes, num_pool,
                                                 num_conv_per_stage,
                                                 feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs,
                                                 dropout_op,
                                                 dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision,
                                                 dropout_in_localization,
                                                 final_nonlin, weightInitializer, pool_op_kernel_sizes,
                                                 conv_kernel_sizes,
                                                 upscale_logits, convolutional_pooling, convolutional_upsampling,
                                                 max_num_features,
                                                 basic_block, seg_output_use_bias, no_skips=no_skips)

        # Re-register the submodules in this order (module and state_dict order).
        conv_blocks_localization = self.conv_blocks_localization
        conv_blocks_context = self.conv_blocks_context
        td = self.td
        tu = self.tu
        seg_outputs = self.seg_outputs
        del self.conv_blocks_localization, self.conv_blocks_context, self.td, self.tu, self.seg_outputs

        self.conv_blocks_context = conv_blocks_context
        self.td = td
        self.tu = tu
        self.conv_blocks_localization = conv_blocks_localization
        self.seg_outputs = seg_outputs
        self.set_sparsity(0.95, 0.975)

    def sparsify_features(self, x: torch.Tensor, during_normal_training: bool = False) -> torch.Tensor:
        """Apply input sparsification (channel-consistent dropout) to a 2D feature map of shape (B, C, H, W).

        during_normal_training=True: augmentation with rate ``sparsity_aug`` on floor(B / 4), at least one,
        randomly chosen samples; the other samples are returned unchanged.
        during_normal_training=False: rate ``sparsity_replay`` on every sample (features stored for replay).

        The mask is redrawn until it is not all zero. Afterwards, ``_apply_sparsity_invariant_normalization``
        normalizes the masked samples if at least 90% of their entries are zero (in the augmentation branch the
        zero fraction is taken over the masked samples only, not over the whole batch).
        """
        if during_normal_training:
            p = self.sparsity_aug
            batch_size = x.shape[0]
            num_to_dropout = torch.div(batch_size, 4, rounding_mode='floor')
            num_to_dropout = torch.maximum(torch.tensor(1), num_to_dropout)

            batch_indices = torch.randperm(batch_size, device=x.device)
            batch_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            batch_mask.index_fill_(0, batch_indices[:num_to_dropout], True)

            # Keep sampling until we get a non-zero mask
            spatial_mask = torch.zeros(batch_size, 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            while torch.all(spatial_mask == 0):
                spatial_mask = torch.bernoulli(
                    torch.ones(batch_size, 1, x.shape[2], x.shape[3],
                               device=x.device, dtype=x.dtype) * (1 - p)
                ) / (1 - p)  # Global normalization

            spatial_mask = spatial_mask.expand(-1, x.shape[1], -1, -1)
            mask_expanded = batch_mask.view(-1, 1, 1, 1).expand_as(x)
            x_sparse = torch.where(mask_expanded, x * spatial_mask, x)

            # Local normalisation on top, for extreme sparsity, decided on the masked samples only
            normalized = x_sparse.clone()
            normalized[batch_mask] = self._apply_sparsity_invariant_normalization(x_sparse[batch_mask],
                                                                                  threshold=0.90)
            x_sparse = normalized
        else:
            p = self.sparsity_replay
            # Keep sampling until we get a non-zero mask
            spatial_mask = torch.zeros_like(x[:, :1, :, :])
            while torch.all(spatial_mask == 0):
                spatial_mask = torch.bernoulli(torch.ones_like(x[:, :1, :, :]) * (1 - p)) / (
                        1 - p)  # Global normalization

            spatial_mask = spatial_mask.expand(-1, x.shape[1], -1, -1)
            x_sparse = x * spatial_mask

            # Local normalisation on top, for extreme sparsity
            x_sparse = self._apply_sparsity_invariant_normalization(x_sparse, threshold=0.90)

        return x_sparse

    def set_sparsity(self, sparsity_aug: float, sparsity_replay: float):
        """Set the channel-consistent dropout (CCD) rates of the two uses of input sparsification.

        sparsity_aug: rate applied as augmentation to a quarter of the current-domain batch during training.
        sparsity_replay: rate applied to the feature maps that are stored for replay.
        """
        for name, value in (('sparsity_aug', sparsity_aug), ('sparsity_replay', sparsity_replay)):
            if not 0 <= value < 1:
                raise ValueError("%s must be in [0, 1), got %r" % (name, value))
        self.sparsity_aug = float(sparsity_aug)
        self.sparsity_replay = float(sparsity_replay)

    def forward(self, x, layer_name_for_feature_extraction: str = None):
        """U-Net forward pass.

        In training mode, input sparsification with ``sparsity_aug`` is applied after every encoder block but
        the bottleneck. If ``layer_name_for_feature_extraction`` is given, the method also returns the list of
        all skip connections up to that layer plus that layer's output, each sparsified with
        ``sparsity_replay``; replay (``feature_forward``) uses only the first entry.
        """
        skips = []
        seg_outputs = []

        # Encoder path
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)

            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                features_and_skips = [self.sparsify_features(skip) for skip in skips]
                features_and_skips.append(self.sparsify_features(x))

            if self.training:
                # During normal training, apply input sparsification to 25% of the samples
                x = self.sparsify_features(x, during_normal_training=True)

            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

            if layer_name_for_feature_extraction == "td." + str(d):
                features_and_skips = [self.sparsify_features(skip) for skip in skips]
                features_and_skips.append(self.sparsify_features(x))

        x = self.conv_blocks_context[-1](x)
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context) - 1):
            features_and_skips = [self.sparsify_features(skip) for skip in skips]
            features_and_skips.append(self.sparsify_features(x))

        # Decoder path
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)

            if layer_name_for_feature_extraction == "tu." + str(u):
                features_and_skips = [self.sparsify_features(skip) for skip in skips]
                features_and_skips.append(self.sparsify_features(x))

            x = self.conv_blocks_localization[u](x)

            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                features_and_skips = [self.sparsify_features(skip) for skip in skips]
                features_and_skips.append(self.sparsify_features(x))

            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            if layer_name_for_feature_extraction is not None:

                return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                                  zip(list(self.upscale_logits_ops)[::-1],
                                                      seg_outputs[:-1][::-1])]), features_and_skips
            else:
                return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                                  zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            if layer_name_for_feature_extraction is not None:
                return seg_outputs[-1], features_and_skips
            else:
                return seg_outputs[-1]

    def train(self, train_mode: bool = True):
        """Set the train mode, keeping the frozen layers (see freeze_layers) in eval mode."""
        super().train(train_mode)
        if not hasattr(self, 'layer_name_for_feature_extraction'):
            return

        layer_name_for_feature_extraction = self.layer_name_for_feature_extraction

        for d in range(len(self.conv_blocks_context) - 1):
            self.conv_blocks_context[d].train(False)
            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                return

            if not self.convolutional_pooling:
                self.td[d].train(False)

            if layer_name_for_feature_extraction == "td." + str(d):
                return

        self.conv_blocks_context[-1].train(False)
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context) - 1):
            return

        for u in range(len(self.tu)):
            self.tu[u].train(False)
            if layer_name_for_feature_extraction == "tu." + str(u):
                return

            self.conv_blocks_localization[u].train(False)
            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                return
        raise ValueError("we cannot end up here. maybe the layer name for feature extraction is wrong " + str(
            layer_name_for_feature_extraction))

    def _apply_sparsity_invariant_normalization(self, x: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
        """
        Apply sparsity-invariant normalization as per Uhrig et al. 2017
        Only applies when the fraction of zeros is at least threshold.
        Each sample is normalized independently; gradients flow through the division (the counts are constants).
        """
        # Check sparsity level
        sparsity = (x == 0).float().mean().item()

        if sparsity < threshold:
            return x  # Not sparse enough to need special handling

        # Apply Uhrig et al.'s sparsity-invariant normalization, all channels at once
        b, c, h, w = x.shape
        with torch.no_grad():
            # Validity mask (o in the paper), one single-channel image per (sample, channel)
            mask = (x.abs() > 1e-8).float().reshape(b * c, 1, h, w)
            # Count valid inputs per 3x3 filter location (denominator in Uhrig's formula), per channel
            valid_count = F.conv2d(mask, torch.ones(1, 1, 3, 3, device=x.device, dtype=mask.dtype),
                                   padding=1).reshape(b, c, h, w)

        # Normalize only where there are valid inputs
        x_normalized = torch.where(
            valid_count > 0,
            # counts are integers: min=1 keeps the unused branch finite under fp16 autocast
            x / torch.clamp(valid_count, min=1),  # Uhrig's normalization
            x  # Keep original where no valid neighbors
        )
        return x_normalized.to(x.dtype)

    def freeze_layers(self, layer_name_for_feature_extraction: str):
        """Freeze all layers up to and including the named one (no gradients, kept in eval mode)."""
        self.layer_name_for_feature_extraction = layer_name_for_feature_extraction
        self.train(self.training)
        for d in range(len(self.conv_blocks_context) - 1):
            self.conv_blocks_context[d].requires_grad_(requires_grad=False)
            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                return

            if not self.convolutional_pooling:
                self.td[d].requires_grad_(requires_grad=False)

            if layer_name_for_feature_extraction == "td." + str(d):
                return

        self.conv_blocks_context[-1].requires_grad_(requires_grad=False)
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context) - 1):
            return

        for u in range(len(self.tu)):
            self.tu[u].requires_grad_(requires_grad=False)
            if layer_name_for_feature_extraction == "tu." + str(u):
                return

            self.conv_blocks_localization[u].requires_grad_(requires_grad=False)
            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                return
        raise ValueError("we cannot end up here. maybe the layer name for feature extraction is wrong " + str(
            layer_name_for_feature_extraction))

    def feature_forward(self, features_and_skips: List[torch.Tensor]):
        """Replay: run the network from stored features. Only the first-level feature map is used."""
        assert hasattr(self, 'layer_name_for_feature_extraction')
        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)

        # Only use the first feature map
        x = features_and_skips[0]
        skips = []
        seg_outputs = []

        # Since we start with the first feature map (already processed by first conv block)
        # We start from index 1 in the encoder path
        if layer == "conv_blocks_context":
            # First feature map is already processed, so start from next block
            skips.append(x)  # Save the first feature map we received
            if not self.convolutional_pooling:
                x = self.td[0](x)

            for d in range(1, id + 1):
                x = self.conv_blocks_context[d](x)
                if d < id:
                    skips.append(x)
                    if not self.convolutional_pooling:
                        x = self.td[d](x)

            if id < len(self.conv_blocks_context) - 1:
                skips.append(x)
                if not self.convolutional_pooling:
                    x = self.td[id](x)

                # Continue with remaining encoder layers
                for d in range(id + 1, len(self.conv_blocks_context) - 1):
                    x = self.conv_blocks_context[d](x)
                    skips.append(x)
                    if not self.convolutional_pooling:
                        x = self.td[d](x)

        elif layer == "td":
            # First feature map is already processed by first conv block
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[0](x)

            for d in range(1, id + 1):
                x = self.conv_blocks_context[d](x)
                skips.append(x)
                if not self.convolutional_pooling and d < id:
                    x = self.td[d](x)

            # Continue with remaining encoder layers
            for d in range(id + 1, len(self.conv_blocks_context) - 1):
                x = self.conv_blocks_context[d](x)
                skips.append(x)
                if not self.convolutional_pooling:
                    x = self.td[d](x)

        # Process bottleneck if needed
        if id < len(self.conv_blocks_context) - 1 and layer in ["td", "conv_blocks_context"]:
            x = self.conv_blocks_context[-1](x)

        # Decoder path
        if layer in ["td", "conv_blocks_context"]:
            for u in range(len(self.tu)):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        elif layer == "conv_blocks_localization":
            # Compute features up to the specified localization block
            for u in range(id):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)

            # Continue with remaining decoder layers
            for u in range(id, len(self.tu)):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        else:  # layer == "tu"
            # Compute features up to the specified upsampling block
            for u in range(id):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)

            if id < len(self.conv_blocks_localization):
                x = self.conv_blocks_localization[id](x)

            # Continue with remaining decoder layers
            for u in range(id + 1, len(self.tu)):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

            if id == len(self.tu) - 1:
                seg_outputs.append(self.final_nonlin(self.seg_outputs[id](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]