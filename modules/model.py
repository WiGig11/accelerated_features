"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from modules.wtconv import WTConv2d

class BasicLayer(nn.Module):
	"""
	Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									nn.BatchNorm2d(out_channels, affine=False),
									nn.ReLU(inplace = True),
									)

	def forward(self, x):
		return self.layer(x)

class BasicWTLayer(nn.Module):
	"""
	Basic Wavelet Convolutional Layer: Wavelet Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									WTConv2d(in_channels, out_channels, padding = padding, kernel_size= kernel_size, stride=stride, bias = bias,wt_levels=2, wt_type='db1'),
									#in_channels, out_channels, padding = 'same', kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'
									#nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									nn.BatchNorm2d(out_channels, affine=False),
									nn.ReLU(inplace = True),
									)

	def forward(self, x):
		return self.layer(x)

class CoorAtten(nn.Module):
    def __init__(self, inp, r=16):
        super().__init__()
        self.h_pool = nn.AdaptiveAvgPool2d((None, 1))  # 新的东西
        self.w_pool = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(inp, inp // r, kernel_size=1, stride=1, padding=0)
        self.norm = nn.BatchNorm2d(inp // r)
        self.nl = nn.ReLU6(inplace=True)
        self.convh = nn.Conv2d(inp // r, inp, kernel_size=1, stride=1, padding=0)
        self.convw = nn.Conv2d(inp // r, inp, kernel_size=1, stride=1, padding=0)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # 无需显式地迁移到设备，所有层已经在初始化时迁移
        b, c, h, w = x.shape
        x_avg = self.h_pool(x)
        y_avg = self.w_pool(x).permute(0, 1, 3, 2).contiguous()
        f = torch.cat([x_avg, y_avg], dim=2)
        f = self.conv1(f)
        f = self.norm(f)
        f = self.nl(f + 3) / 6
        x_f, y_f = torch.split(f, [h, w], dim=2)
        y_f = y_f.permute(0, 1, 3, 2).contiguous()
        x_f = self.sig(self.convh(x_f))
        y_f = self.sig(self.convw(y_f))
        
        x_f = x_f.expand(-1, -1, h, w)
        y_f = y_f.expand(-1, -1, h, w)
        y = x * x_f * y_f
        return y

class PixelShuffleExample(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(PixelShuffleExample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * upscale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
    
class FeatureFusionAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(FeatureFusionAttention, self).__init__()
        self.pooler11 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pooler12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooler21 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pooler22 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算中间通道数
        hidden_channels = in_channels * 4 // reduction
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 4, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(hidden_channels // 4, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        # 不使用 PixelShuffle，而使用插值上采样
        self.uper = None

    def forward(self, w1, w2):
        pooled_flow11 = self.pooler11(w1)
        pooled_flow12 = self.pooler12(w1)
        pooled_flow21 = self.pooler21(w2)
        pooled_flow22 = self.pooler22(w2)
        
        feature = torch.cat([pooled_flow22, pooled_flow21, pooled_flow12, pooled_flow11], dim=1)  # [batch, C*4, H/2, W/2]
        feature = self.conv1(feature)  # [batch, C//reduction, H/2, W/2]
        feature = self.conv2(feature)  # [batch, C//(reduction*4), H/2, W/2]
        feature = self.outconv(feature)  # [batch, C, H/2, W/2]
        alpha = feature  # [batch, C, H/2, W/2]
        alpha = F.interpolate(alpha, scale_factor=2, mode='bilinear', align_corners=False)  # [batch, C, H, W]
        
        # 可选：监控 alpha 的分布
        #if self.training:
        #    alpha_mean = alpha.mean().item()
        #    alpha_std = alpha.std().item()
        #    print(f'Alpha Mean: {alpha_mean:.4f}, Alpha Std: {alpha_std:.4f}')
        
        return w1 * alpha + w2 * (1 - alpha)

class MultiScaleFeatureFusionDensePyramid(nn.Module):
    def __init__(self):
        super(MultiScaleFeatureFusionDensePyramid, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=5e-2)
        )
        self.af4_5 = FeatureFusionAttention(in_channels=64)
        self.af45_3 = FeatureFusionAttention(in_channels=64)
        self.acti1 = nn.LeakyReLU(negative_slope=5e-2)
        self.acti2 = nn.LeakyReLU(negative_slope=5e-2)
        self.acti3 = nn.LeakyReLU(negative_slope=5e-2)
        
        # 初始化权重
        self._initialize_weights()

    def forward(self, x3, x4, x5):
        feats45_prime = self.af4_5(self.conv1(x4), self.conv2(x5))  # 融合 x4 和 x5
        res = self.af45_3(feats45_prime, self.conv3(x3))  # 融合 feats45_prime 和 x3
        return res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class XFeatModel(nn.Module):
	"""
	   Implementation of architecture described in 
	   "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	"""

	def __init__(self,coora = True,fusion = True,wtconv = True):
		super().__init__()
		self.fusion = fusion
		self.wtconv = wtconv
		self.norm = nn.InstanceNorm2d(1)


		########### ⬇️ CNN Backbone & Heads ⬇️ ###########
		self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
									nn.Conv2d (1, 24, 1, stride = 1, padding=0) )

		self.block1 = nn.Sequential(
										BasicLayer( 1,  4, stride=1),
										BasicLayer( 4,  8, stride=2),
										BasicLayer( 8,  8, stride=1),
										BasicLayer( 8, 24, stride=2),
									)
		if coora==True:
			self.block2 = nn.Sequential(
											CoorAtten( 24, r=8),
											BasicLayer(24, 24, stride=1),
											BasicLayer(24, 24, stride=1),
										)

			self.block3 = nn.Sequential(
											CoorAtten( 24, r=8),
											BasicLayer(24, 64, stride=2),
											BasicLayer(64, 64, stride=1),
											BasicLayer(64, 64, 1, padding=0),
										)
			self.block4 = nn.Sequential(
											CoorAtten( 64, r=16),
											BasicLayer(64, 64, stride=2),
											BasicLayer(64, 64, stride=1),
											BasicLayer(64, 64, stride=1),
										)

			self.block5 = nn.Sequential(
											CoorAtten(  64, r=16),
											BasicLayer( 64, 128, stride=2),
											BasicLayer(128, 128, stride=1),
											BasicLayer(128, 128, stride=1),
											BasicLayer(128,  64, 1, padding=0),
										)
		else:
			self.block2 = nn.Sequential(
											BasicLayer(24, 24, stride=1),
											BasicLayer(24, 24, stride=1),
										)

			self.block3 = nn.Sequential(
											BasicLayer(24, 64, stride=2),
											BasicLayer(64, 64, stride=1),
											BasicLayer(64, 64, 1, padding=0),
										)
			self.block4 = nn.Sequential(
											BasicLayer(64, 64, stride=2),
											BasicLayer(64, 64, stride=1),
											BasicLayer(64, 64, stride=1),
										)

			self.block5 = nn.Sequential(
											BasicLayer( 64, 128, stride=2),
											BasicLayer(128, 128, stride=1),
											BasicLayer(128, 128, stride=1),
											BasicLayer(128,  64, 1, padding=0),
										)

		self.block_fusion =  nn.Sequential(
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, stride=1),
										nn.Conv2d (64, 64, 1, padding=0)
									 )

		self.heatmap_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 1, 1),
										nn.Sigmoid()
									)

		if self.wtconv == True:
			self.keypoint_head = nn.Sequential(
											BasicWTLayer(64, 64, 1, padding=0),
											BasicWTLayer(64, 64, 1, padding=0),
											BasicWTLayer(64, 64, 1, padding=0),
											nn.Conv2d (64, 65, 1),
										)
		else:
			self.keypoint_head = nn.Sequential(
											BasicLayer(64, 64, 1, padding=0),
											BasicLayer(64, 64, 1, padding=0),
											BasicLayer(64, 64, 1, padding=0),
											nn.Conv2d (64, 65, 1),
										)


  		########### ⬇️ Fine Matcher MLP ⬇️ ###########

		self.fine_matcher =  nn.Sequential(
											nn.Linear(128, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 64),
										)
		if fusion==True:
			self.fusioner = MultiScaleFeatureFusionDensePyramid()

	def _unfold2d(self, x, ws = 2):
		"""
			Unfolds tensor in 2D with desired ws (window size) and concat the channels
		"""
		B, C, H, W = x.shape
		x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
			.reshape(B, C, H//ws, W//ws, ws**2)
		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


	def forward(self, x):
		"""
			input:
				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
			return:
				feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
				keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
				heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

		"""
		#dont backprop through normalization
		with torch.no_grad():
			x = x.mean(dim=1, keepdim = True)
			x = self.norm(x)

		#main backbone
		x1 = self.block1(x)
		x2 = self.block2(x1 + self.skip1(x))
		x3 = self.block3(x2)
		x4 = self.block4(x3)
		x5 = self.block5(x4)

		#pyramid fusion
		x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
		x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
		if self.fusion:
			feats = self.fusioner(x3,x4,x5)
		else:
			feats = (x3+x4+x5)
		feats = self.block_fusion(feats)

		#heads
		heatmap = self.heatmap_head(feats) # Reliability map
		keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits

		return feats, keypoints, heatmap
