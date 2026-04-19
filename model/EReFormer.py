import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from torch.nn import init
from .vit import trunc_normal_
from .swin import SwinTransformer, SwinBlock, PatchSplitting, CrossSwinBlock
from .submodules import GRViT
from operator import mul
from functools import reduce
from .model_util import *
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
# 添加 dinov2 源码路径
dinov2_path = "/home/server/tianrui/dinov2"  # 替换为你的实际路径
sys.path.append(dinov2_path)
# 导入模型定义
from dinov2.dinov2.models.vision_transformer import vit_large

import torch
import torch.nn.functional as F


def calculate_kernel_mi(feat1, feat2, sigma=1.0):
    x = F.normalize(feat1, dim=-1)
    y = F.normalize(feat2, dim=-1)
    # 4. 计算核矩阵
    dist_sq_x = torch.cdist(x, x, p=2) ** 2
    dist_sq_y = torch.cdist(y, y, p=2) ** 2
    sigma_x = dist_sq_x.mean() + 1e-9
    sigma_y = dist_sq_y.mean() + 1e-9
    Kx = torch.exp(-dist_sq_x / (2 * sigma_x))
    Ky = torch.exp(-dist_sq_y / (2 * sigma_y))
    # 5. 计算相似度
    mi_score = torch.mean(Kx * Ky)
    return mi_score


# # --- 测试 ---
# feat1 = torch.randn(2, 96, 56, 56)
# feat2 = torch.randn(2, 96, 1, 1)
#
# # 直接调用，函数内部会自动处理尺寸
# score = calculate_kernel_mi(feat1, feat2)
# print(f"核互信息分数: {score.item():.4f}")



def calculate_covariance_score(feat1, feat2):
    """计算两个特征图的平均协方差"""
    similarity = F.cosine_similarity(feat1, feat2, dim=-1)
    return similarity.mean()




class FiLMFusion(nn.Module):
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        # 文本特征生成视觉特征的缩放(gamma)和偏移(beta)
        self.gamma_layer = nn.Linear(text_dim, visual_dim)
        self.beta_layer = nn.Linear(text_dim, visual_dim)
        # 投影层，将文本映射到视觉维度进行残差连接
    def forward(self, vis, txt):
        # 1. 生成FiLM参数
        gamma = self.gamma_layer(txt)  # (bs, 1, C_v)
        beta = self.beta_layer(txt)  # (bs, 1, C_v)
        # 2. 调制视觉特征 (融合发生)
        # 这里的公式是: output = gamma * vis + beta
        # 文本特征直接控制了视觉特征的分布
        # 3. 残差连接 (引入文本信息)
        # 将映射后的文本特征加到视觉特征上
        out = gamma * vis + beta + txt
        return out


class TransformerRecurrent(nn.Module):
    """
    
    """
    def __init__(self, EReFormer_kwargs):
        super(TransformerRecurrent, self).__init__()
        
        self.embed_dim = EReFormer_kwargs['embed_dim']
        self.window_size = EReFormer_kwargs['window_size']
        self.img_size = EReFormer_kwargs['img_size']
        self.patch_size = EReFormer_kwargs['patch_size']
        self.in_chans = EReFormer_kwargs['in_chans']
        self.depth = EReFormer_kwargs['depth']
        self.num_heads = EReFormer_kwargs['num_heads']
        self.upsampling_method = EReFormer_kwargs['upsampling_method']
        self.pretrained = EReFormer_kwargs['pretrained'] 
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = self.norm_layer(self.embed_dim)
        self.num_output_channels = 1
        # self.features=[192, 384, 768, 1536]
        self.features=[96, 192, 384, 768]
        decoder_drop = 0.1


        self.proj_layers_0 = nn.Linear(768, self.features[0])
        self.proj_layers_1 = nn.Linear(768, self.features[1])
        self.proj_layers_2 = nn.Linear(768, self.features[2])
        self.proj_layers_3 = nn.Linear(768, self.features[3])
        self.fusion1 = FiLMFusion(96, 96)
        self.fusion2 = FiLMFusion(192, 192)
        self.fusion3 = FiLMFusion(384, 384)
        self.fusion4 = FiLMFusion(768, 768)

        # new add ######################################################################
        self.learnable_tokens = nn.Parameter(
            torch.empty([100, 384]))  # gang zhu shi
        #
        self.mlp_token2feat = nn.Linear(384, 384)  # gang zhu shi
        self.mlp_delta_f = nn.Linear(384, 384)  # gang zhu shi
        #
        #
        #
        val = math.sqrt(6.0 / float(3 * reduce(mul, (16, 16), 1) + 384))  # gang zhu shi
        nn.init.uniform_(self.learnable_tokens.data, -val, val)  # gang zhu shi
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))  # gang zhu shi
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))  # gang zhu shi



        self.model_dinov2 = vit_large(
            patch_size=14, img_size=224, init_values=1.0,
            ffn_layer="mlp", block_chunks=0, num_register_tokens=0,
            interpolate_antialias=False, interpolate_offset=0.1, )
        # 加载预训练权重
        ckpt_path = "/home/server/tianrui/DepthForge/checkpoints/dinov2_vitl14_pretrain.pth"  # 替换为你的权重路径
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # 获取新 grid size (224 / 14 = 16)
        # new_grid_size = 224 // 14  # = 16
        # 替换 pos_embed
        ckpt["pos_embed"] = self.interpolate_pos_encoding(ckpt["pos_embed"],
                                                          new_grid_size=16, old_grid_size=37)  # 因为 518/14=37

        # 4. 加载权重（现在 pos_embed 匹配了！）
        self.model_dinov2.load_state_dict(ckpt, strict=True)
        self.model_dinov2 = self.model_dinov2.to(device)
        # 冻结模型
        self.model_dinov2.eval()
        for param in self.model_dinov2.parameters():
            param.requires_grad = False
        print("✅ DINOv2 ViT-L/14 loaded successfully (offline)!")
        # 在 __init__ 中（加载 DINOv2 之后）
        self.dino_target_layers = [7, 11, 15, 23]
        self.features_dinov2 = {}  # 用于存储 hook 输出

        # 注册 hooks（只注册一次，在 __init__ 中）
        def get_hook(name):
            def hook(module, input, output):
                self.features_dinov2[name] = output[:, 1:, :]  # (B, L, C)

            return hook

        for idx in self.dino_target_layers:
            self.model_dinov2.blocks[idx].register_forward_hook(get_hook(f'layer_{idx}'))

        self.spatial_proj = nn.Conv2d(1024, 384, kernel_size=1)
        #### new add #############################################################


        #encoder
        self.SwinTransformer = SwinTransformer(embed_dims=self.embed_dim, depths=self.depth, \
                                               num_heads=self.num_heads, window_size=self.window_size, \
                                                   pretrained=self.pretrained)
        
        #GRViT
        self.GRViT_0 = GRViT(dim=self.features[0], num_heads=self.num_heads[0], mlp_ratio=2., qkv_bias=False, \
                           drop=decoder_drop, attn_drop=0., proj_drop=0.)
        self.GRViT_1 = GRViT(dim=self.features[1], num_heads=self.num_heads[1], mlp_ratio=2., qkv_bias=False, \
                           drop=decoder_drop, attn_drop=0., proj_drop=0.)
        self.GRViT_2 = GRViT(dim=self.features[2], num_heads=self.num_heads[2], mlp_ratio=2., qkv_bias=False, \
                           drop=decoder_drop, attn_drop=0., proj_drop=0.)
        self.GRViT_3 = GRViT(dim=self.features[3], num_heads=self.num_heads[3], mlp_ratio=2., qkv_bias=False, \
                           drop=decoder_drop, attn_drop=0., proj_drop=0.)
            
        self.pos_embed_0 = nn.Parameter(torch.zeros(
            1, (224 // 4)*(224 // 4), self.features[0]))
        self.pos_embed_1 = nn.Parameter(torch.zeros(
            1, (224 // 8)*(224 // 8), self.features[1]))
        self.pos_embed_2 = nn.Parameter(torch.zeros(
            1, (224 // 16)*(224 // 16), self.features[2]))
        self.pos_embed_3 = nn.Parameter(torch.zeros(
            1, (224 // 32)*(224 // 32), self.features[3]))
        
        trunc_normal_(self.pos_embed_0, std=.02)
        trunc_normal_(self.pos_embed_1, std=.02)
        trunc_normal_(self.pos_embed_2, std=.02)
        trunc_normal_(self.pos_embed_3, std=.02)
        
        #res    
        self.resblock_0 = SwinBlock(embed_dims=self.features[3], num_heads=self.num_heads[3], \
                                         feedforward_channels=2 * self.features[3], window_size=self.window_size,\
                                         shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)  
        self.resblock_1 = SwinBlock(embed_dims=self.features[3], num_heads=self.num_heads[3], \
                                         feedforward_channels=2 * self.features[3], window_size=self.window_size, \
                                         shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        
        #decoder1
        self.decoder1_block0 = SwinBlock(embed_dims=self.features[3], num_heads=self.num_heads[3], \
                                         feedforward_channels=2 * self.features[3], window_size=self.window_size,\
                                         shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
        self.decoder1_block1 = SwinBlock(embed_dims=self.features[3], num_heads=self.num_heads[3], \
                                         feedforward_channels=2 * self.features[3], window_size=self.window_size, \
                                         shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        
        self.PatchSplitting1 = PatchSplitting(in_channels=self.features[3], out_channels=self.features[2],\
                                             output_size=(self.img_size[0]//16, self.img_size[1]//16), stride=2)
        
        #STF1
        self.decoder2_query0 = CrossSwinBlock(embed_dims=self.features[2], num_heads=self.num_heads[2], \
                                             feedforward_channels=4 * self.features[2], window_size=self.window_size,\
                                             shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        self.decoder2_query1 = CrossSwinBlock(embed_dims=self.features[2], num_heads=self.num_heads[2], \
                                             feedforward_channels=4 * self.features[2], window_size=self.window_size,\
                                             shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        
        #decoder2
        self.decoder2_block0 = SwinBlock(embed_dims=self.features[2], num_heads=self.num_heads[2], \
                                         feedforward_channels=4 * self.features[2], window_size=self.window_size,\
                                         shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
        self.decoder2_block1 = SwinBlock(embed_dims=self.features[2], num_heads=self.num_heads[2], \
                                         feedforward_channels=4 * self.features[2], window_size=self.window_size,\
                                         shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
            
        self.PatchSplitting2 = PatchSplitting(in_channels=self.features[2], out_channels=self.features[1],\
                                             output_size=(self.img_size[0]//8, self.img_size[1]//8), stride=2)
        
        #STF2
        self.decoder3_query0 = CrossSwinBlock(embed_dims=self.features[1], num_heads=self.num_heads[1], \
                                             feedforward_channels=4 * self.features[1], window_size=self.window_size,\
                                             shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        self.decoder3_query1 = CrossSwinBlock(embed_dims=self.features[1], num_heads=self.num_heads[1], \
                                             feedforward_channels=4 * self.features[1], window_size=self.window_size,\
                                             shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
         
        #decoder3
        self.decoder3_block0 = SwinBlock(embed_dims=self.features[1], num_heads=self.num_heads[1], \
                                         feedforward_channels=4 * self.features[1], window_size=self.window_size,\
                                         shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
        self.decoder3_block1 = SwinBlock(embed_dims=self.features[1], num_heads=self.num_heads[1], \
                                         feedforward_channels=4 * self.features[1], window_size=self.window_size,\
                                         shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
            
        self.PatchSplitting3 = PatchSplitting(in_channels=self.features[1], out_channels=self.features[0],\
                            output_size=(int(math.ceil(self.img_size[0] / 4)), int(math.ceil(self.img_size[1] / 4))), stride=2)
            
        #STF3
        self.decoder4_query0 = CrossSwinBlock(embed_dims=self.features[0], num_heads=self.num_heads[0], \
                                             feedforward_channels=4 * self.features[0], window_size=self.window_size,\
                                             shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
        self.decoder4_query1 = CrossSwinBlock(embed_dims=self.features[0], num_heads=self.num_heads[0], \
                                             feedforward_channels=4 * self.features[0], window_size=self.window_size,\
                                             shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.)
         
        #decoder4
        self.decoder4_block0 = SwinBlock(embed_dims=self.features[0], num_heads=self.num_heads[0], \
                                         feedforward_channels=4 * self.features[0], window_size=self.window_size,\
                                         shift=False, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
        self.decoder4_block1 = SwinBlock(embed_dims=self.features[0], num_heads=self.num_heads[0], \
                                         feedforward_channels=4 * self.features[0], window_size=self.window_size,\
                                         shift=True, drop_rate=decoder_drop, attn_drop_rate=0., drop_path_rate=0.) 
            
        self.Unflatten = nn.Unflatten(2, torch.Size([int(math.ceil(self.img_size[0] / self.patch_size)),\
                                                     int(math.ceil(self.img_size[1] / self.patch_size))]))  
        
        
        #head
        self.conv2d0 = nn.Conv2d(self.features[0], 96, kernel_size=3, stride=1, padding=1)
        self.head = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
              
        self.num_decoders = EReFormer_kwargs['num_decoders']
        self.states = [None] * self.num_decoders

    def interpolate_pos_encoding(self, pos_embed, new_grid_size, old_grid_size=37):
        """
        将 [1, 1 + old^2, C] 插值为 [1, 1 + new^2, C]

        """
        if new_grid_size == old_grid_size:
            return pos_embed
        cls_token = pos_embed[:, :1, :]  # (1, 1, C)
        pos_tokens = pos_embed[:, 1:, :]  # (1, old^2, C)
        # 转为 2D: (1, C, old, old)
        dim = pos_tokens.shape[-1]
        pos_tokens = pos_tokens.reshape(1, old_grid_size, old_grid_size, dim).permute(0, 3, 1, 2)
        # 插值到 new_grid_size
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens,
            size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
        # 转回 1D: (1, new^2, C)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, -1, dim)
        # 拼接 cls token
        new_pos_embed = torch.cat([cls_token, pos_tokens], dim=1)
        return new_pos_embed
    
    def _resize_pos_embed(self, posemb, gs_h, gs_w):
        # posemb_tok, posemb_grid = (
        #     posemb[:, : 1],
        #     posemb[0, 1 :],
        # )
        posemb_grid = posemb

        # gs_old = int(math.sqrt(len(posemb_grid)))
        gs_old = int(math.sqrt(posemb_grid.shape[1]))

        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        
        posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bicubic")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        # posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        posemb = posemb_grid

        return posemb
        

    def forward(self, x, rgb_image, text_features):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """
        # vit encoder
        B, _, H, W = x.shape
        text_features = text_features.to('cuda')
        text_features1 = self.proj_layers_0(text_features)
        text_features2 = self.proj_layers_1(text_features)
        text_features3 = self.proj_layers_2(text_features)
        text_features4 = self.proj_layers_3(text_features)
        #
        # self.features.clear()  # 清空前一次结果
        with torch.no_grad():
            _ = self.model_dinov2(rgb_image)
        # feat7 = self.features_dinov2['layer_7']
        # feat11 = self.features_dinov2['layer_11']
        feat15 = self.features_dinov2['layer_15']
        # #feat23 = self.features_dinov2['layer_23']
        # # dino_feats = [feat7, feat11, feat15, feat23]
        # # Step 1: DINOv2 → 2D feature map (16x16)
        feat15 = feat15.transpose(1, 2).view(B, 1024, 16, 16)
        # Step 2: Resize to 14x14 (match Swin Stage3 spatial size)
        feat15 = F.interpolate(feat15, size=(14, 14), mode='bilinear', align_corners=False)
        # Step 3: Project channel 1024 → 384
        feat15 = self.spatial_proj(feat15)  # (B, 384, 14, 14)
        # Step 4: Back to token form
        feat15 = feat15.flatten(2).transpose(1, 2)  # (B, 196, 384)



        stage_feature = self.SwinTransformer(x, self.learnable_tokens, feat15, self.mlp_token2feat, self.mlp_delta_f)
        
        stage1 = stage_feature[0]
        stage2 = stage_feature[1]
        stage3 = stage_feature[2]
        stage4 = stage_feature[-1] #B, H/32*W/32, 8C

        # print("self.swtich1(stage1, text_features1)::::", self.swtich1(stage1, text_features1))
        sim_1 = calculate_covariance_score(stage1, text_features1)
        sim_2 = calculate_covariance_score(stage2, text_features2)
        sim_3 = calculate_covariance_score(stage3, text_features3)
        sim_4 = calculate_covariance_score(stage4, text_features4)
        print("simsimsimsim:::::", sim_1, "***", sim_2,"***", sim_3, "***", sim_4)

        mi_1 = calculate_kernel_mi(stage1, text_features1)
        mi_2 = calculate_kernel_mi(stage2, text_features2)
        mi_3 = calculate_kernel_mi(stage3, text_features3)
        mi_4 = calculate_kernel_mi(stage4, text_features4)
        print("mimimimimimimimimi:::::", mi_1, "***", mi_2, "***", mi_3, "***", mi_4)

        # simsimsimsim::::: tensor(-0.0475, device='cuda:0', grad_fn= < MeanBackward0 >) ** *tensor(-0.0724,
        #                                                                                           device='cuda:0',
        #                                                                                           grad_fn= < MeanBackward0 >) ** *tensor(
        #     -0.0477, device='cuda:0', grad_fn= < MeanBackward0 >) ** *tensor(-0.0159, device='cuda:0',
        #                                                                      grad_fn= < MeanBackward0 >)
        # mimimimimimimimimi::::: tensor(0.7562, device='cuda:0', grad_fn= < MeanBackward0 >) ** *tensor(0.6690,
        #                                                                                                device='cuda:0',
        #                                                                                                grad_fn= < MeanBackward0 >) ** *tensor(
        #     0.6255, device='cuda:0', grad_fn= < MeanBackward0 >) ** *tensor(0.6539, device='cuda:0',
        #                                                                     grad_fn= < MeanBackward0 >)

        if (abs(sim_1) <0.2) and (abs(mi_1)<0.8):
            print("zouleme")
            stage1 = self.fusion1(stage1, text_features1)



        if (abs(sim_2) < 0.2) and (abs(mi_2) < 0.8):
            print("zouleme")
            stage2 = self.fusion2(stage2, text_features2)


        if (abs(sim_3) < 0.2) and (abs(mi_3) < 0.8):
            print("zouleme")
            stage3 = self.fusion3(stage3, text_features3)


        if (abs(sim_4) < 0.2) and (abs(mi_4) < 0.8):
            print("zouleme")
            stage4 = self.fusion4(stage4, text_features4)













        
        pos_embed_0 = self._resize_pos_embed(
            self.pos_embed_0, int(math.ceil(H / 4)), int(math.ceil(W / 4))
        )
        pos_embed_1 = self._resize_pos_embed(
            self.pos_embed_1, int(math.ceil(H / 8)), int(math.ceil(W / 8))
        )
        pos_embed_2 = self._resize_pos_embed(
            self.pos_embed_2, int(math.ceil(H / 16)), int(math.ceil(W / 16))
        )
        pos_embed_3 = self._resize_pos_embed(
            self.pos_embed_3, int(math.ceil(H / 32)), int(math.ceil(W / 32))
        )
        
        #GRViT
        stage1, states_0 = self.GRViT_0(stage1, self.states[0], pos_embed_0)
        self.states[0] = states_0
        stage2, states_1 = self.GRViT_1(stage2, self.states[1], pos_embed_1)
        self.states[1] = states_1
        stage3, states_2 = self.GRViT_2(stage3, self.states[2], pos_embed_2)
        self.states[2] = states_2
        stage4, states_3 = self.GRViT_3(stage4, self.states[3], pos_embed_3)
        self.states[3] = states_3
        
        #res
        hw_shape = (int(math.ceil(H / 32)), int(math.ceil(W / 32)))
        res = self.resblock_0(stage4, hw_shape)
        res = self.resblock_1(res, hw_shape) #B, H/32*W/32, 8C
        
        #decoder1 
        fuse_stage4 = res + stage4
        hw_shape = (int(math.ceil(H / 32)), int(math.ceil(W / 32)))
        decoder_stage4 = self.decoder1_block0(fuse_stage4, hw_shape)
        decoder_stage4 = self.decoder1_block1(decoder_stage4, hw_shape) #B, H/32*W/32, 8C
        up_decoder_stage4 = self.PatchSplitting1(decoder_stage4) #B, H/16*W/16, 4C
        
        #STF1
        hw_shape = (int(math.ceil(H / 16)), int(math.ceil(W / 16)))
        query_stage3 = self.decoder2_query0(up_decoder_stage4, stage3, hw_shape)
        query_stage3 = self.decoder2_query1(query_stage3, stage3, hw_shape)
        fuse_stage3 = up_decoder_stage4 + query_stage3
        
        #decoder2
        decoder_stage3 = self.decoder2_block0(fuse_stage3, hw_shape)
        decoder_stage3 = self.decoder2_block1(decoder_stage3, hw_shape) #B, H/16*W/16, 4C
        up_decoder_stage3 = self.PatchSplitting2(decoder_stage3) #B, H/8*W/8, 2C
        
        #STF2 
        hw_shape = (int(math.ceil(H / 8)), int(math.ceil(W / 8)))
        query_stage2 = self.decoder3_query0(up_decoder_stage3, stage2, hw_shape)
        query_stage2 = self.decoder3_query1(query_stage2, stage2, hw_shape)
        fuse_stage2 = up_decoder_stage3 + query_stage2
        
        #decoder3
        hw_shape = (int(math.ceil(H / 8)), int(math.ceil(W / 8)))
        decoder_stage2 = self.decoder3_block0(fuse_stage2, hw_shape)
        decoder_stage2 = self.decoder3_block1(decoder_stage2, hw_shape) #B, H/8*W/8, 2C
        up_decoder_stage2 = self.PatchSplitting3(decoder_stage2) #B, H/4*W/4, C
        
        #STF3 
        hw_shape = (int(math.ceil(H / 4)), int(math.ceil(W / 4)))
        query_stage1 = self.decoder4_query0(up_decoder_stage2, stage1, hw_shape)
        query_stage1 = self.decoder4_query1(query_stage1, stage1, hw_shape)
        fuse_stage1 = up_decoder_stage2 + query_stage1
        
        #decoder4
        hw_shape = (int(math.ceil(H / 4)), int(math.ceil(W / 4)))
        decoder_stage1 = self.decoder4_block0(fuse_stage1, hw_shape)
        decoder_stage1 = self.decoder4_block1(decoder_stage1, hw_shape) #B, H/4*W/4, C
        
        decoder = self.norm(decoder_stage1)
        decoder = decoder.transpose(1, 2)
        decoder = self.Unflatten(decoder) #B, C, H/4, W/4
            
        out = self.conv2d0(decoder)
        out = F.interpolate(out, size=(self.img_size[0], self.img_size[1]), mode='bicubic', align_corners=True)
        # out = F.interpolate(out, size=(256, 336), mode='bicubic', align_corners=True)
        depth = self.head(out)
        # print(depth.shape)
                    
        return { 'pred_depth':depth}
