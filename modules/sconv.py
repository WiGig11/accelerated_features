import torch
import torch.nn as nn


# 定义修正后的 SpatialConvModule
class SpatialConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SpatialConvModule, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 定义四个方向的卷积层
        self.conv_D = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.conv_U = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.conv_R = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.conv_L = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

        # 1x1 卷积用于融合四个方向的特征
        self.fusion_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward_direction(self, x, conv, direction):
        """
        在指定方向上进行特征传播
        direction: 'D', 'U', 'R', 'L'
        """
        B, C, H, W = x.size()
        if direction == 'D':
            # 从上到下
            res = torch.zeros(B, C, 1, W).to(x.device)  # 初始化为零，形状为 (B, C, 1, W)
            out = []
            for i in range(H):
                current = x[:, :, i, :].unsqueeze(2)  # (B, C, 1, W)
                res = conv(current + res)
                out.append(res)
            out = torch.cat(out, dim=2)  # 拼接高度维度
            return out

        elif direction == 'U':
            # 从下到上
            res = torch.zeros(B, C, 1, W).to(x.device)  # 初始化为零，形状为 (B, C, 1, W)
            out = []
            for i in reversed(range(H)):
                current = x[:, :, i, :].unsqueeze(2)
                res = conv(current + res)
                out.append(res)
            out = torch.cat(out[::-1], dim=2)  # 反转后拼接
            return out

        elif direction == 'R':
            # 从左到右
            res = torch.zeros(B, C, H, 1).to(x.device)  # 初始化为零，形状为 (B, C, H, 1)
            out = []
            for i in range(W):
                current = x[:, :, :, i].unsqueeze(3)  # (B, C, H, 1)
                res = conv(current + res)
                out.append(res)
            out = torch.cat(out, dim=3)  # 拼接宽度维度
            return out

        elif direction == 'L':
            # 从右到左
            res = torch.zeros(B, C, H, 1).to(x.device)  # 初始化为零，形状为 (B, C, H, 1)
            out = []
            for i in reversed(range(W)):
                current = x[:, :, :, i].unsqueeze(3)
                res = conv(current + res)
                out.append(res)
            out = torch.cat(out[::-1], dim=3)  # 反转后拼接
            return out
        else:
            raise ValueError("Unsupported direction")

    def forward(self, x):
        # 四个方向的特征
        # 创建四个不同的 CUDA 流
        stream_D = torch.cuda.Stream()
        stream_U = torch.cuda.Stream()
        stream_R = torch.cuda.Stream()
        stream_L = torch.cuda.Stream()
        with torch.cuda.stream(stream_D):
            feat_D = self.forward_direction(x, self.conv_D, 'D')
            
        with torch.cuda.stream(stream_U):
            feat_U = self.forward_direction(x, self.conv_U, 'U')

        with torch.cuda.stream(stream_R):
            feat_R = self.forward_direction(x, self.conv_R, 'R')

        with torch.cuda.stream(stream_L):
            feat_L = self.forward_direction(x, self.conv_L, 'L')

        # 确保所有流的计算完成后再进行下一步
        torch.cuda.synchronize()
        
        # 拼接四个方向的特征
        feat = torch.cat([feat_D, feat_U, feat_R, feat_L], dim=1)  # 在通道维度拼接

        # 融合特征
        out = self.fusion_conv(feat)
        out = self.bn(out)
        out = self.relu(out)

        return out

'''
# 定义输入参数
batch_size = 1
in_channels = 1
out_channels = 1
height = 64
width = 64

# 创建虚拟输入张量
dummy_input = torch.randn(batch_size, in_channels, height, width)

# 实例化 SpatialConvModule
spatial_conv = SpatialConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

# 将虚拟输入传递给模块
output = spatial_conv(dummy_input)

# 打印输出的形状
print(f"输入形状: {dummy_input.shape}")
print(f"输出形状: {output.shape}")
'''
