from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F

from ...utils import point_sample
from ..prompt_decoder import build_prompt_encoder
from .position_encoding import PositionEmbeddingRandom
from ..utils import (
    MLP,
    CrossAttentionLayer,
    FFNLayer,
    SelfAttentionLayer,
)


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""


def build_transformer_decoder(cfg):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.OVSEG.TRANSFORMER_DECODER_NAME
    in_channels = cfg.MODEL.OVSEG.CONVS_DIM
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels)


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        *,
        embed_dim: int,
        prompt_encoder: nn.Module,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        # positional encoding
        N_steps = embed_dim // 2
        # self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.pe_layer = PositionEmbeddingRandom(N_steps)

        # define Transformer decoder here
        self.mask_dim = mask_dim
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.tgt_mask = None
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=embed_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=embed_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=embed_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        # extra self-attention layer for learnable query features
        self.extra_self_attention_layer = SelfAttentionLayer(
            d_model=embed_dim,
            nhead=nheads,
            dropout=0.0,
            normalize_before=pre_norm,
        )
        self.feat_embed = nn.Conv2d(embed_dim, mask_dim, 1, bias=True)

        # visual prompt encoder
        self.prompt_encoder = prompt_encoder

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, embed_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != embed_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, embed_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # mask branch
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.mask_embed = nn.ModuleList(
            [MLP(embed_dim, embed_dim, mask_dim, 3) for _ in range(5)]
        )
        self.iou_embed = nn.ModuleList(
            [MLP(embed_dim, embed_dim, 1, 3) for _ in range(5)]
        )

    @classmethod
    def from_config(cls, cfg, in_channels):
        ret = {}
        ret["in_channels"] = in_channels
        ret["embed_dim"] = cfg.MODEL.OVSEG.EMBED_DIM
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.OVSEG.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.OVSEG.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.OVSEG.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.OVSEG.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.OVSEG.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.OVSEG.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.OVSEG.MASK_DIM
        ret["prompt_encoder"] = build_prompt_encoder(cfg)

        return ret

    def get_center(self, points: List[torch.Tensor]):
        centers = []
        for point in points:
            center = (point[..., 2:] + point[..., :2]) / 2
            center[..., 0] /= self.prompt_encoder.image_size[1]
            center[..., 1] /= self.prompt_encoder.image_size[0]
            centers.append(center)
        centers = torch.stack(centers, dim=0)
        return centers

    def sample_center(self, centers: torch.Tensor, mask_features: torch.Tensor):
        if centers.dim() == 4:
            N, Q, K, _ = centers.shape
            centers = centers.reshape(N, Q * K, 2)
            feature_c = point_sample(mask_features, centers, align_corners=False)
            feature_c = feature_c.permute(0, 2, 1).reshape(N, Q, K, -1)  # N, Q, K, C
        else:
            feature_c = point_sample(mask_features, centers, align_corners=False)
            feature_c = feature_c.permute(0, 2, 1)  # N, Q, C
            K = 2
        return feature_c, K

    def forward(
        self,
        x: List[torch.Tensor],
        mask_features: torch.Tensor,
        points: List[torch.Tensor] = None,
        boxes: List[torch.Tensor] = None,
        points_multi: List[torch.Tensor] = None,
    ):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src, pos, size_list = [], [], []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            # flatten NxCxHxW to HWxNxC
            pos.append(self.pe_layer(x[i], None).flatten(2).permute(2, 0, 1))
            src.append(
                self.input_proj[i](x[i]).flatten(2).permute(2, 0, 1)
                + self.level_embed.weight[i][None, None, :]
            )
            # # flatten NxCxHxW to NxHWxC
            # pos.append(self.pe_layer(x[i], None).flatten(2).permute(0, 2, 1))
            # src.append(
            #     self.input_proj[i](x[i]).flatten(2).permute(0, 2, 1)
            #     + self.level_embed.weight[i][None, None, :]
            # )

        # calculate centers of points (points is a box with the xyxy format).
        # N, Q, 2 / N, Q, K, 2
        if points is not None:
            # Q, 2
            points_c = self.get_center(points)
        if boxes is not None:
            # Q, 2
            points_c = self.get_center(boxes)
        if points_multi is not None:
            # Q, K, 2
            points_c = self.get_center(points_multi[0])
        # sample feature vectors corresponding to centers by grid sample
        # N, Q, K, C / N, Q, C
        feature_c, K_ = self.sample_center(points_c, mask_features)

        output, query_embed = self.prompt_encoder(
            points, boxes, points_multi, feature_c
        )  # N, Q, K, C
        N, Q, K, C = output.shape

        # prediction heads on learnable query features
        predictions_mask = []
        predictions_iou = []
        output = output.reshape(N * Q, K, C).permute(1, 0, 2)
        output = self.extra_self_attention_layer(
            output,
            query_pos=None,
            tgt_mask=None,
        )
        output = output.permute(1, 0, 2).reshape(N, Q, K, C)
        outputs_mask, outputs_iou, attn_mask = self.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=size_list[0],
            multimask=points is not None,
            num_prompt=K_,
        )
        predictions_mask.append(outputs_mask)
        predictions_iou.append(outputs_iou)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = output.reshape(N, Q * K, C).permute(1, 0, 2)
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed.reshape(N, Q * K, C).permute(1, 0, 2),  # QK, N, C
            )
            output = output.reshape(Q, K, N, C).permute(1, 2, 0, 3).reshape(K, N * Q, C)
            # output = output.reshape(N, Q, K, C).reshape(N * Q, K, C)
            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed.reshape(N * Q, K, C).permute(1, 0, 2),
            )
            output = self.transformer_ffn_layers[i](output)
            output = output.permute(1, 0, 2).reshape(N, Q, K, C)
            # output = output.reshape(N, Q, K, C)

            (
                outputs_mask,
                outputs_iou,
                attn_mask,
            ) = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                multimask=points is not None,
                num_prompt=K_,
            )
            predictions_mask.append(outputs_mask)
            predictions_iou.append(outputs_iou)

        assert len(predictions_mask) == self.num_layers + 1

        out = {
            "pred_masks": predictions_mask[-1],
            "pred_ious": predictions_iou[-1],
            "aux_outputs": self._set_aux_loss(predictions_mask, predictions_iou),
        }
        return out

    def forward_prediction_heads(
        self,
        output,
        mask_features,
        attn_mask_target_size,
        multimask=False,
        num_prompt=None,
    ):
        N, Q, K, _ = output.shape
        _, C, H, W = mask_features.shape
        mask_features = self.feat_embed(mask_features)
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output[:, :, num_prompt:, :]

        if multimask:
            K_ = K - (num_prompt + 1)
            decoder_output = decoder_output[:, :, 1:, :]
            mask_embed = torch.stack(
                [self.mask_embed[i + 1](decoder_output[:, :, i]) for i in range(K_)],
                dim=2,
            )
            iou_embed = torch.stack(
                [self.iou_embed[i + 1](decoder_output[:, :, i]) for i in range(K_)],
                dim=2,
            )
        else:
            K_ = 1
            mask_embed = self.mask_embed[0](decoder_output[:, :, :1, :])
            iou_embed = self.iou_embed[0](decoder_output[:, :, :1, :])

        # mask branch
        outputs_mask = torch.einsum("bqkc,bchw->bqkhw", mask_embed, mask_features)
        outputs_iou = iou_embed.squeeze(-1).sigmoid()

        # NOTE: prediction is of higher-resolution
        # [B, Q, K, H, W] -> [B, QK, H, W] -> [B, Q, K, H*W]
        attn_mask = outputs_mask.reshape(N, Q * K_, H, W)
        attn_mask = F.interpolate(
            attn_mask,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        ).reshape(N, Q, K_, attn_mask_target_size[0], attn_mask_target_size[1])

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        # [B, Q, K, H, W] -> [B, Q, 1, H, W] -> [B, 1, Q, 1, H, W] -> [B, 1, Q, 1, H*W] -> [B, h, Q, K, H*W] -> [B*h, Q*K, H*W]
        attn_mask = (
            attn_mask.sigmoid()
            .ge(0.5)
            .sum(dim=2, keepdim=True)
            .bool()
            .unsqueeze(1)
            .flatten(-2)
            .repeat(1, self.num_heads, 1, K, 1)
            .reshape(
                N * self.num_heads,
                Q * K,
                attn_mask_target_size[0] * attn_mask_target_size[1],
            )
        ).detach()
        attn_mask = ~attn_mask

        return outputs_mask, outputs_iou, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, predictions_mask, predictions_iou):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_masks": mask, "pred_ious": iou}
            for mask, iou in zip(predictions_mask[:-1], predictions_iou[:-1])
        ]
