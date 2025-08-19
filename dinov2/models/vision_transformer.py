# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from dinov2.layers import (
    Mlp,
    PatchEmbed,
    PatchEmbed3D,
    SwiGLUFFNFused,
    AttentionQKVSplit,
    MemEffAttention,
    MemEffAttentionQKVSplit,
    NestedTensorBlock as Block
)
from timm.layers import SwiGLU as SwiGLU_timm, RotaryEmbeddingCat
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd, PatchDecode

logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x, rope=None):
        for b in self:
            if isinstance(b, nn.Identity):
                x = b(x)
            else:
                x = b(x, rope=rope)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        norm_in_attn=False,
        norm_in_ffn=False,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        pos_embed_for_register_tokens=False,
        use_rope=False,
        use_cls=True,
        rope_ref_shape=224,
        mode="2D",
        up_projection=False
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.mode = mode
        self.pos_embed_for_register_tokens = pos_embed_for_register_tokens
        self.use_rope = use_rope
        self.use_cls = use_cls
        self.num_tokens = (1 if self.use_cls else 0) + (num_register_tokens if pos_embed_for_register_tokens else 0)
        self.norm_in_attn = norm_in_attn
        self.norm_in_ffn = norm_in_ffn
        self.rope_ref_shape = rope_ref_shape

        print("Using cls token: ", self.use_cls)

        if self.mode == "3D":
            embed_layer = PatchEmbed3D
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        
        if use_rope:
            rope_dim = round(embed_dim // num_heads / 1.5)
            assert rope_dim == embed_dim / num_heads / 1.5, "rope dim must be divsible by (num_heads * 1.5)"
            assert rope_dim % 4 == 0, "rope dim must be divisible by 4"
            ref_feat_shape = (img_size // patch_size, img_size // patch_size, img_size // patch_size)
            self.rope = RotaryEmbeddingCat(
                rope_dim, in_pixels=False, feat_shape=None, ref_feat_shape=(self.rope_ref_shape, self.rope_ref_shape, self.rope_ref_shape)
            )
        else:
            self.rope = None

        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "swiglu_timm":
            ffn_layer = SwiGLU_timm
            act_layer = nn.SiLU
            logger.info("using SwiGLU layer from timm as FFN")
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                norm_in_attn=norm_in_attn,
                norm_in_ffn=norm_in_ffn,
                num_prefix_tokens=self.num_tokens
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        
        self.up_projection = None
        if up_projection:
            self.up_projection = PatchDecode(
                (8, 8, 8), embed_dim, 2, norm=LayerNormNd, activation=nn.GELU
            )

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _interpolate_pos_encoding_2d(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def _interpolate_pos_encoding_3d(self, x, d, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1] - self.num_tokens
        N = self.pos_embed.shape[1] - self.num_tokens
        if npatch == N and h == w == d:
            return self.pos_embed
        
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, self.num_tokens:]
        reg_pos_embed = pos_embed[:, 1:self.num_tokens] if \
            self.pos_embed_for_register_tokens and (self.register_tokens is not None) else None
        tokens_pos_embed = torch.cat((class_pos_embed.unsqueeze(0), reg_pos_embed), dim=1) \
            if reg_pos_embed is not None else class_pos_embed.unsqueeze(0)
        if not self.use_cls:
            tokens_pos_embed = tokens_pos_embed[:, 1:, :]
        dim = x.shape[-1]
        h0 = h // self.patch_size
        w0 = w // self.patch_size
        d0 = d // self.patch_size

        M = int(math.ceil(N ** (1/3)))  # Recover the number of patches in each dimension
        assert N == M * M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            sz = float(d0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sz, sy, sx)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (d0, h0, w0)
        patch_pos_embed = nn.functional.interpolate(
            # patch_pos_embed.reshape(1, M, M, M, dim).permute(0, 4, 3, 1, 2),
            patch_pos_embed.reshape(1, dim, M, M, M),
            mode="trilinear",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        # print("BEFORE ASSERTION", (d0, h0, w0), patch_pos_embed.shape[-3:])
        assert (d0, h0, w0) == patch_pos_embed.shape[-3:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return torch.cat((tokens_pos_embed, patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        if self.mode == "2D":
            B, nc, w, h = x.shape
        else:
            B, nc, d, h, w = x.shape # d, w, h ???
        
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        tokens = self.cls_token.expand(x.shape[0], -1, -1) \
            if (not self.pos_embed_for_register_tokens) or (self.register_tokens is None) \
            else torch.cat((self.cls_token.expand(x.shape[0], -1, -1), self.register_tokens.expand(x.shape[0], -1, -1)), dim=1)
        if not self.use_cls:
            tokens = None # tokens[:, 1:, :]

        x = torch.cat((tokens, x), dim=1) if tokens is not None else x
        if self.mode == "2D":
            x = x + self._interpolate_pos_encoding_2d(x, w, h)
        else:
            x = x + self._interpolate_pos_encoding_3d(x, d, h, w)

        if (not self.pos_embed_for_register_tokens) and (self.register_tokens is not None):
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        shape = (d // self.patch_size, h // self.patch_size, w // self.patch_size)
        rot_pos_emb = self.rope.get_embed(shape=shape) if self.use_rope else None
        return x, rot_pos_emb

    def forward_features_list(self, x_list, masks_list):
        if masks_list is None:
            masks_list = [None] * len(x_list)
        x, rot_pos_emb = zip(*[self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)])
        for blk in self.blocks:
            x = blk(list(x), rope=rot_pos_emb)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            cls_present = 1 if self.use_cls else 0
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0] if self.use_cls else x_norm.mean(dim=1, keepdim=False),
                    "x_norm_regtokens": x_norm[:, cls_present : self.num_register_tokens + cls_present],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + cls_present:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x, rot_pos_emb = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x, rope=rot_pos_emb)

        x_norm = self.norm(x)
        cls_present = 1 if self.use_cls else 0
        return {
            "x_norm_clstoken": x_norm[:, 0] if self.use_cls else x_norm.mean(dim=1, keepdim=False),
            "x_norm_regtokens": x_norm[:, cls_present : self.num_register_tokens + cls_present],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + cls_present :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=True, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if self.up_projection is not None:
            import numpy as np
            prod_n_patches = np.prod(ret["x_norm_patchtokens"].shape)
            bs = ret["x_norm_patchtokens"].shape[0]
            n_patches_dim = int(round((prod_n_patches.item() // (self.embed_dim * bs)) ** (1 / 3)))
            return self.up_projection(ret["x_norm_patchtokens"].movedim(1, 2).reshape(-1, self.embed_dim, n_patches_dim, n_patches_dim, n_patches_dim))
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, split_qkv=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttentionQKVSplit if split_qkv else MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_medium(patch_size=16, num_register_tokens=0, split_qkv=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=864,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttentionQKVSplit if split_qkv else MemEffAttention),#
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, split_qkv=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttentionQKVSplit if split_qkv else MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, split_qkv=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttentionQKVSplit if split_qkv else MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, split_qkv=False, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttentionQKVSplit if split_qkv else MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_small_3d(patch_size=16, num_register_tokens=0, split_qkv=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttentionQKVSplit if split_qkv else MemEffAttention),
        num_register_tokens=num_register_tokens,
        mode="3D",
        in_chans=1,
        embed_layer=PatchEmbed3D,
        **kwargs,
    )
    return model


def vit_base_3d(patch_size=16, num_register_tokens=0, split_qkv=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttentionQKVSplit if split_qkv else MemEffAttention),
        num_register_tokens=num_register_tokens,
        mode="3D",
        in_chans=1,
        embed_layer=PatchEmbed3D,
        **kwargs,
    )
    return model


def vit_medium_3d(patch_size=16, num_register_tokens=0, split_qkv=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=864,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttentionQKVSplit if split_qkv else MemEffAttention),
        num_register_tokens=num_register_tokens,
        mode="3D",
        in_chans=1,
        embed_layer=PatchEmbed3D,
        **kwargs,
    )
    return model


def vit_large_3d(patch_size=16, num_register_tokens=0, split_qkv=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttentionQKVSplit if split_qkv else MemEffAttention),
        num_register_tokens=num_register_tokens,
        mode="3D",
        in_chans=1,
        embed_layer=PatchEmbed3D,
        **kwargs,
    )
    return model


def vit_giant2_3d(patch_size=16, num_register_tokens=0, split_qkv=False, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttentionQKVSplit if split_qkv else MemEffAttention),
        num_register_tokens=num_register_tokens,
        mode="3D",
        in_chans=1,
        embed_layer=PatchEmbed3D,
        **kwargs,
    )
    return model
