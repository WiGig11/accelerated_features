"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""

import numpy as np
import os
import torch
import tqdm

from modules.model import XFeatModel

os.environ['CUDA_VISIBLE_DEVICES'] = 'cuda:1' #Force CPU, comment for GPU

xfeat = XFeatModel(coora = False,fusion = False)

#Random input
x = torch.randn(1,3,480,640)

feats, keypoints, heatmap = xfeat(x)
print("feats shape==============",feats.shape)
print("keypoints shape==========",keypoints.shape)
print("heatmap shape============",heatmap.shape)