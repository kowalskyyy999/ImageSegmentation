#!/usr/bin/python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import crop_image

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding = 1, stride = 1, dilation = 1, K = 1):
        super(SeparableConv, self).__init__()
        self.separable = nn.Sequential(
            nn.Conv2d(in_channels, 
                      K * in_channels, 
                      kernel_size = 3, 
                      padding = padding, 
                      stride=stride, 
                      dilation=dilation, 
                      groups=in_channels),
            nn.BatchNorm2d(K * in_channels),
            nn.ReLU(),
            nn.Conv2d(K * in_channels, out_channels, kernel_size = 1, padding=0)
        )
    def forward(self, x):
        return self.separable(x)
    
class EntryFlow(nn.Module):
    def __init__(self, input_channels, init_filter):
        super(EntryFlow, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, init_filter, kernel_size = 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(init_filter, init_filter*2, kernel_size = 3, padding=1)
        self.relu = nn.ReLU()
        
        self.block1 = self.makeBlockSeparable(init_filter*2, init_filter*4)
        self.resConv1 = nn.Conv2d(init_filter*2, init_filter*4, kernel_size = 1, stride=2)
        
        self.block2 = self.makeBlockSeparable(init_filter*8, init_filter*8)
        self.resConv2 = nn.Conv2d(init_filter*4, init_filter*8, kernel_size = 1, stride=2)
        
        self.block3 = self.makeBlockSeparable(init_filter*16, init_filter*16)
        self.resConv3 = nn.Conv2d(init_filter*8, init_filter*16, kernel_size = 1, stride = 2)

        self.fc = nn.Conv2d(int(init_filter * 32), int(init_filter * 22.75), kernel_size=1)
        
    def makeBlockSeparable(self, in_channels, out_channels):
        block = nn.Sequential(
            SeparableConv(in_channels, out_channels),
            SeparableConv(out_channels, out_channels),
            SeparableConv(out_channels, out_channels, stride=2)
        )
        
        return block
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        out_block1 = self.block1(x)
        out_res1 = self.resConv1(x)
        x = torch.cat((out_block1, out_res1), dim=1)
        
        out = x
        
        out_block2 = self.block2(x)
        out_res2 = self.resConv2(out_res1)
        x = torch.cat((out_block2, out_res2), dim=1)
        
        out_block3 = self.block3(x)
        out_res3 = self.resConv3(out_res2)
        x = torch.cat((out_block3, out_res3), dim=1)

        x = self.fc(x)

        return x, out
    
class MiddleFlow(nn.Module):
    def __init__(self, init_filter):
        super(MiddleFlow, self).__init__()
        filter = int(init_filter * 22.75)
        self.block = nn.Sequential(
            SeparableConv(filter, filter, padding = 1),
            SeparableConv(filter, filter, padding = 1),
            SeparableConv(filter, filter, padding = 1)
        )
        self.last = nn.Conv2d(2 * filter, filter, kernel_size = 1)
        
    def forward(self, x):
        out = self.block(x)
        out = torch.cat((out, x), dim=1)
        out = self.last(out)
        return out

class ExitFlow(nn.Module):
    def __init__(self, init_filter):
        super(ExitFlow, self).__init__()
        self.block1 = nn.Sequential(
            SeparableConv(int(init_filter*22.75), int(init_filter*22.75), padding = 1),
            SeparableConv(int(init_filter*22.75), int(init_filter*32), padding = 1),
            SeparableConv(int(init_filter*32), int(init_filter*32), padding = 1, stride = 2)
        )
        self.block2 = nn.Sequential(
            SeparableConv(int(2 * 32 * init_filter), int(init_filter * 48), padding = 1),
            SeparableConv(int(init_filter * 48), int(init_filter * 48), padding = 1),
            SeparableConv(int(init_filter * 48), int(init_filter *2 * 32), padding = 1)
        )
        self.resConv = nn.Conv2d(int(init_filter*22.75), int(init_filter * 32), kernel_size = 1, stride=2)
        
    def forward(self, x):
        out_block = self.block1(x)
        out_res = self.resConv(x)
        x = torch.cat((out_block, out_res), dim=1)
        x = self.block2(x)
        return x
    
    
class ASPP(nn.Module):
    def __init__(self, in_channels, img_size):
        super(ASPP, self).__init__()
        h = int(np.ceil(img_size[0]/16))
        w = int(np.ceil(img_size[1]/16))
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size = 1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, dilation = 6, padding = 6)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, dilation = 12, padding = 12)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, dilation = 18, padding = 18)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(h, w))
        self.fc = nn.Conv2d(5 * in_channels, int(in_channels/2), kernel_size = 1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv(self.conv(self.conv2(x)))
        x3 = self.conv(self.conv(self.conv3(x)))
        x4 = self.conv(self.conv(self.conv4(x)))
        x5 = self.pooling(x)
        x = torch.cat((x1, x2, x3, x4, x5), dim = 1)
        out = self.fc(x)
        return out
    
    
class Encoder(nn.Module):
    def __init__(self, in_channels, init_filter, img_size):
        super(Encoder, self).__init__()
        self.entry = EntryFlow(in_channels, init_filter)
        self.middle = self.IterableLayer(init_filter)
        self.depthAtrousConv = SeparableConv(int(init_filter * 22.75), init_filter * 16, padding = 2, dilation =2, stride = 1)
        self.aspp = ASPP(init_filter*16, img_size)
    
    def IterableLayer(self, init_filter):
        layers = []
        for _ in range(16):
            layers += [MiddleFlow(init_filter)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape : N, 3, H, W
        x, outFeat = self.entry(x)
        # outFeat : N, init_filter*8, H/4, W/16
        x = self.middle(x)
        x = self.depthAtrousConv(x)
        x = self.aspp(x) 
        # x shape : N, init_filter*8, H/16, W/16
        return x, outFeat
    
class Decoder(nn.Module):
    def __init__(self, init_filter):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor = 4, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(init_filter*8, init_filter*8, kernel_size = 1, padding = 0)
        self.conv3 = nn.Conv2d(init_filter*16, init_filter*2, kernel_size = 3)
        self.bn = nn.BatchNorm2d(init_filter*2)
        
    def forward(self, x, outFeat):
        x = self.upsample(x)
        outFeat = self.conv1(outFeat)
        
        x = crop_image(x, outFeat.shape[2:])
        
        x = torch.cat((x, outFeat), dim=1)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.upsample(x)
        return x
    
    
class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels, out_channels, init_filter, img_size):
        super(DeepLabV3Plus, self).__init__()

        self.encoder = Encoder(in_channels, init_filter, img_size)
        self.decoder = Decoder(init_filter)
        self.fc = nn.Sequential(
            nn.Conv2d(init_filter * 2, init_filter, kernel_size=3),
            nn.BatchNorm2d(init_filter),
            nn.ReLU(),
            nn.Conv2d(init_filter, out_channels , kernel_size = 1)
        )
        
    def forward(self, image):
        x, outFeat = self.encoder(image)
        x = self.decoder(x, outFeat)
        x = self.fc(x) 
        
        return x 

# def crop_image(layer, target_size):
#     _, _, h, w = layer.size()
#     y = (h - target_size[0])//2
#     x = (w - target_size[1])//2
#     return layer[:,:, y : (y+target_size[0]), x:(x+target_size[1])] 

def test(img_size):
    x = torch.rand(1, 3, img_size[0], img_size[1])
    model = DeepLabV3Plus(3, 35, 32, img_size)
    out, _, _ = model(x)
    print('Output Shape', out.shape)


