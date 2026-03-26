from torch import nn
from torch.nn import Parameter
import torch
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
            
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s (t: expansion, c: output channels, n: repeat, s: stride)
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

class MobileFaceNet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting, embedding_size=128):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        
        # Explicit block list to ensure consistent naming across layers
        layers = []
        for t, c, n, s in bottleneck_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(block(self.inplanes, c, stride, t))
                self.inplanes = c
        self.blocks = nn.Sequential(*layers)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, (7, 6), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, embedding_size, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x

def get_model(embedding_size=128):
    """Standard entry point for both sides to get the identical architecture."""
    return MobileFaceNet(embedding_size=embedding_size)

def verify_architecture_match(model):
    """Fail-fast check: counts parameters to identify version mismatch early."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ARCH] Total parameters in backbone: {total_params}")
    return total_params

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m_base = m # Rename to m_base for clarity
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        # Precomputed values for default m
        self.update_margin(torch.full((out_features,), m))

    def update_margin(self, margins_tensor):
        """Dynamic update for adaptive margin."""
        self.register_buffer('m', margins_tensor)
        self.register_buffer('cos_m', torch.cos(margins_tensor))
        self.register_buffer('sin_m', torch.sin(margins_tensor))
        self.register_buffer('th', torch.cos(math.pi - margins_tensor))
        self.register_buffer('mm', torch.sin(math.pi - margins_tensor) * margins_tensor)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        eps = 1e-6
        cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # Select margins for the current batch
        batch_cos_m = self.cos_m[label].view(-1, 1) if label.size(0) > 0 else self.cos_m[0]
        batch_sin_m = self.sin_m[label].view(-1, 1) if label.size(0) > 0 else self.sin_m[0]
        batch_th = self.th[label].view(-1, 1) if label.size(0) > 0 else self.th[0]
        batch_mm = self.mm[label].view(-1, 1) if label.size(0) > 0 else self.mm[0]
        batch_m = self.m[label].view(-1, 1) if label.size(0) > 0 else self.m[0]

        phi = cosine * batch_cos_m - sine * batch_sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > batch_th, phi, cosine - batch_mm)
            
        if label.size(0) == 0:
            return torch.zeros(cosine.size(), device=input.device)
            
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output