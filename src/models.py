import torch
from torch import nn
import torch.nn.functional as F
from pytorch_model_summary import summary # pip install pytorch-model-summary



def double_conv(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=False),
        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=False),
        # torch.nn.Dropout()
    )   

def double_deconv(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=False),
        torch.nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=False),
        # torch.nn.Dropout()
    )   


class UNet(torch.nn.Module):
    def __init__(self, nfilter=64):
        super(UNet,self).__init__()
        self.nfilter = nfilter

        self.dconv_down1 = double_conv(1, self.nfilter)
        self.dconv_down2 = double_conv(self.nfilter, self.nfilter*2)
        self.dconv_down3 = double_conv(self.nfilter*2, self.nfilter*4)
        self.dconv_down4 = double_conv(self.nfilter*4, self.nfilter*8)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(self.nfilter*4 + self.nfilter*8, self.nfilter*4)
        self.dconv_up2 = double_conv(self.nfilter*2 + self.nfilter*4, self.nfilter*2)
        self.dconv_up1 = double_conv(self.nfilter*2 + self.nfilter, self.nfilter)
        
        self.conv_last = torch.nn.Conv2d(self.nfilter, 1, 1)

        # self.dconv_down1 = double_conv(1, self.nfilter)
        # self.dconv_down2 = double_conv(self.nfilter, self.nfilter)
        # self.dconv_down3 = double_conv(self.nfilter, self.nfilter)
        # self.dconv_down4 = double_conv(self.nfilter, self.nfilter)        

        # self.maxpool = torch.nn.MaxPool2d(2)
        # self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        # self.dconv_up3 = double_conv(self.nfilter + self.nfilter, self.nfilter)
        # self.dconv_up2 = double_conv(self.nfilter + self.nfilter, self.nfilter)
        # self.dconv_up1 = double_conv(self.nfilter + self.nfilter, self.nfilter)
        
        # self.conv_last = torch.nn.Conv2d(self.nfilter, 1, 1)
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        out = torch.nn.ReLU(inplace=True)(out)
        
        return out

    def summary(self, n=64):
        print("Unet network:")
        if next(self.parameters()).is_cuda:
            print(summary(self, torch.zeros((1,1,n,n)).cuda(), show_input=True))
        else:
            print(summary(self, torch.zeros((1,1,n,n)), show_input=True))