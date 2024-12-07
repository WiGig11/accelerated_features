"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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


class FeatureFusionAttention(nn.Module):
    def __init__(self):
        super(FeatureFusionAttention, self).__init__()
        self.pooler11 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.pooler12 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pooler21 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.pooler22 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(256, 64, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=4, stride=2, padding=0)
        self.linear = None  # 初始化为 None
    
    def forward(self, w1, w2):
        pooled_flow11 = self.pooler11(w1)
        pooled_flow12 = self.pooler12(w1)
        pooled_flow21 = self.pooler21(w2)
        pooled_flow22 = self.pooler22(w2)
        feature = torch.cat([pooled_flow22, pooled_flow21, pooled_flow12, pooled_flow11], dim=1)
        feature = self.conv2(self.conv1(feature))  # 输出形状为 [batch_size, 16, H, W]
        b, _, _, _ = feature.shape
        feature = feature.view(b, -1)  # 展平为 [batch_size, 16 * H * W]
        
        if self.linear is None:
            # 动态创建线性层
            self.linear = self.build_linear(feature.shape[1], 1).to(feature.device)
            # 手动注册参数
            self.add_module('linear', self.linear)
        
        feature = self.linear(feature)
        alpha = torch.sigmoid(feature)  # 形状为 [batch_size, 1]
        alpha = alpha.view(alpha.size(0), 1, 1, 1)  # 扩展为 [batch_size, 1, 1, 1]
        return w1 * alpha + w2 * (1 - alpha)
    
    def build_linear(self, inp, oup):
        if inp > 256:
            outp1 = 256
            layer = nn.Sequential(
                nn.Linear(inp, outp1),
                nn.ReLU(),
                nn.Linear(outp1, oup)
            )
        else:
            layer = nn.Linear(inp, oup)
        return layer

        

class MultiScaleFeatureFusionDensePyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64,64,kernel_size = 1,stride = 1,padding  = 0)
        self.conv2 = nn.Conv2d(64,64,kernel_size = 1,stride = 1,padding  = 0)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 1,stride = 1,padding  = 0)
        self.af4_5 = FeatureFusionAttention()
        self.af45_3 = FeatureFusionAttention()

    def forward(self,x3,x4,x5):
        feats45_prime = self.af4_5(self.conv1(x4),self.conv2(x5))
        res = self.af45_3(feats45_prime,self.conv3(x3))
        return res


class XFeatModel(nn.Module):
	"""
	   Implementation of architecture described in 
	   "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	"""

	def __init__(self,coora = True,fusion = True):
		super().__init__()
		self.fusion = fusion
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
