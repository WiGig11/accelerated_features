import torch
import torch.nn as nn
import pywt
import pywt.data
import torch.nn.functional as F
from functools import partial

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:,:,0,:,:]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x



# *! test only
class ExampleModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ExampleModel, self).__init__()
        self.wt_conv = WTConv2d(in_channels, in_channels, kernel_size=3, stride=1, wt_levels=2, wt_type='db1')
        self.relu = nn.ReLU()
        #self.fc = nn.Linear(in_channels * 64 * 64, num_classes)  # 假设特征图尺寸为 32x32

    def forward(self, x):
        x = self.wt_conv(x)
        #print(x.shape)
        x = self.relu(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)
        return x

class ExampleModel2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ExampleModel2, self).__init__()
        self.wt_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        #self.fc = nn.Linear(in_channels * 64 * 64, num_classes)  # 假设特征图尺寸为 32x32

    def forward(self, x):
        x = self.wt_conv(x)
        #print(x.shape)
        x = self.relu(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)
        return x
# 模型实例化
model = ExampleModel(in_channels=3, num_classes=10)
model2 = ExampleModel2(in_channels=3, num_classes=10)
import time

input_tensor = torch.randn(1, 3, 64, 64)  # 示例输入
t1 = time.time()
output = model(input_tensor)
t2 = time.time()
print(output.shape)  # 输出形状
print(t2-t1)  # 输出形状

t1 = time.time()
output = model2(input_tensor)
t2 = time.time()
print(output.shape)  # 输出形状
print(t2-t1)  # 输出形状



'''


import torch
import torch.nn as nn

# 假设 CoorAtten 和 BasicLayer 已定义
class CoorAtten(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        # 定义层
    def forward(self, x):
        return x

class BasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.conv(x))

# Model1定义
class Model1(nn.Module):
    """
       test only
    """
    def __init__(self, coora=True):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)
        if coora:
            self.block2 = nn.Sequential(
                
                BasicLayer(1, 24, stride=1),
                BasicLayer(24, 24, stride=1),
            )
        self.coora = coora
    def forward(self, x):
        x = self.norm(x)
        if self.coora:
            x = self.block2(x)
        return x

# Model2定义
class Model2(nn.Module):
    """
       test only
    """
    def __init__(self, coora=True, fusion=False):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)
        if coora:
            self.block2 = nn.Sequential(
                
                BasicLayer(1, 24, stride=1),
                BasicLayer(24, 24, stride=1),
            )
        if fusion:
            self.block3 = nn.Sequential(
                
                BasicLayer(24, 24, stride=1),
                BasicLayer(24, 24, stride=1),
            )
        self.coora = coora
        self.fusion = fusion
    def forward(self, x):
        x = self.norm(x)
        if self.coora:
            x = self.block2(x)
        if self.fusion:
            x = self.block3(x)
        return x

# 训练 Model1 并保存权重
model1 = Model1(coora=True)
# ... 训练过程 ...
torch.save(model1.state_dict(), 'model1_weights.pth')

# 定义 Model2 并加载 Model1 的权重
model2 = Model2(coora=True, fusion=True)
state_dict = torch.load('model1_weights.pth')
model2.load_state_dict(state_dict, strict=True)

# 验证加载是否成功
model2.eval()
with torch.no_grad():
    dummy_input = torch.randn(1, 1, 64, 64)  # 示例输入
    output = model2(dummy_input)
    print(output.shape)
'''



