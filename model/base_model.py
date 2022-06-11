from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torchvision import models

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_d(nn.Module):
    def __init__(self, d_ch, out_ch):
        super(conv_block_d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(d_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, y):
        y = self.conv(y)
        return y

class Linear_tensor32(nn.Module):
    def __init__(self):
        super(Linear_tensor32,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(32, 2), requires_grad = True)
        torch.nn.init.constant_(self.weight,1.0)


    def forward(self, x,y):
        weight = F.softmax(self.weight)
        return weight[ :,0].unsqueeze(0).unsqueeze(2).unsqueeze(3) * x+ weight[ :,1].unsqueeze(0).unsqueeze(2).unsqueeze(3)*y

class Linear_tensor64(nn.Module):
    def __init__(self):
        super(Linear_tensor64,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(64, 2), requires_grad = True)
        torch.nn.init.constant_(self.weight,1.0)

    def forward(self, x,y):
        weight = F.softmax(self.weight)
        return weight[ :,0].unsqueeze(0).unsqueeze(2).unsqueeze(3) * x+ weight[ :,1].unsqueeze(0).unsqueeze(2).unsqueeze(3)*y

class Linear_tensor128(nn.Module):
    def __init__(self):
        super(Linear_tensor128,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(128, 2), requires_grad = True)
        torch.nn.init.constant_(self.weight,1.0)


    def forward(self, x,y):
        weight = F.softmax(self.weight)
        return weight[ :,0].unsqueeze(0).unsqueeze(2).unsqueeze(3) * x+ weight[ :,1].unsqueeze(0).unsqueeze(2).unsqueeze(3)*y

class Linear_tensor256(nn.Module):
    def __init__(self):
        super(Linear_tensor256,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(256, 2), requires_grad = True)
        torch.nn.init.constant_(self.weight,1.0)


    def forward(self, x,y):
        weight = F.softmax(self.weight)
        return weight[ :,0].unsqueeze(0).unsqueeze(2).unsqueeze(3) * x+ weight[ :,1].unsqueeze(0).unsqueeze(2).unsqueeze(3)*y

class Linear_tensor512(nn.Module):
    def __init__(self):
        super(Linear_tensor512,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(512, 2), requires_grad = True)
        torch.nn.init.constant_(self.weight,1.0)


    def forward(self, x,y):
        weight = F.softmax(self.weight)
        return weight[ :,0].unsqueeze(0).unsqueeze(2).unsqueeze(3) * x+ weight[ :,1].unsqueeze(0).unsqueeze(2).unsqueeze(3)*y

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class MSA_filter(nn.Module):
    """
    MSA_filter: using channel attention
    to filter noise of feature from encoder
    """
    def __init__(self, in_ch):
        super(MSA_filter, self).__init__()

        self.conv_rgb = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_dep = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

        self.filter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 8, in_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, rgb, depth):
        mid_rgb = self.conv_rgb(rgb)
        mid_depth = self.conv_dep(depth)
        rad = mid_rgb + mid_depth
        channel_map = self.filter(rad)
        new_rgb = rgb * channel_map
        new_depth = depth * channel_map
        new_rad = new_rgb + new_depth
        return new_rad

class MSA_HL(nn.Module):
    """
    HL: highlight the structure of feature map
    """
    def __init__(self, in_ch):
        super(MSA_HL, self).__init__()
        self.conv_earlier = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_later = nn.Sequential(
             nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
             nn.BatchNorm2d(in_ch),
             nn.ReLU(inplace=True)
         )
        self.hl = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, rad, x):
        x_avg = torch.mean(rad, dim=1, keepdim=True)
        x_max, _ = torch.max(rad, dim=1, keepdim=True)
        x_avg_max = torch.cat((x_avg, x_max), dim=1)
        spatial_map = rad * self.hl(x_avg_max)

        mid_x = self.conv_earlier(x)
        new_x = mid_x + spatial_map
        out = self.conv_later(new_x)
        return out

class MSA(nn.Module):
    """
    MSA: Multi-source attention
    """
    def __init__(self, in_ch):
        super(MSA, self).__init__()

        self.channel_map = MSA_filter(in_ch)
        self.spatial_map = MSA_HL(in_ch)

    def forward(self, rgb, depth, x):
        new_rad = self.channel_map(rgb, depth)
        out = self.spatial_map(new_rad, x)
        return out


############Encoder###########
class Encoder(nn.Module):
    def __init__(self, in_ch):
        super(Encoder, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])  # donnot change the size but reduce the channel
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

    def forward(self, x):
        e1 = self.Conv1(x)  # batchsize*32*512*512

        e2 = self.Maxpool1(e1)  # 32*256*256
        e2 = self.Conv2(e2)  # 64*256*256

        e3 = self.Maxpool2(e2)  # 64*128*128
        e3 = self.Conv3(e3)  # 128*128*128

        e4 = self.Maxpool3(e3)  # 128*64*64
        e4 = self.Conv4(e4)  # 256*64*64

        e5 = self.Maxpool4(e4)  # 256*32*32
        e5 = self.Conv5(e5)  # 512*32*32

        return e1, e2, e3, e4, e5

class DualEncoder(nn.Module):
    def __init__(self, in_ch=3, d_ch=1):
        super(DualEncoder, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # rgb encoder
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        # depth encoder
        self.Maxpool1_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5_d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_d = conv_block_d(d_ch, filters[0])
        self.Conv2_d = conv_block_d(filters[0], filters[1])
        self.Conv3_d = conv_block_d(filters[1], filters[2])
        self.Conv4_d = conv_block_d(filters[2], filters[3])
        self.Conv5_d = conv_block_d(filters[3], filters[4])

    def forward(self, x, y):
        e1 = self.Conv1(x)  # batchsize*64*512*512
        t1 = self.Conv1_d(y)  # 64*512*512
        e1 = e1 + t1  # fuse 64*1024*1024

        e2 = self.Maxpool1(e1)  # 64*256*256
        e2 = self.Conv2(e2)  # 128*256*256
        t2 = self.Maxpool1_d(t1)  # 64*256*256
        t2 = self.Conv2_d(t2)  # 128*256*256
        e2 = e2 + t2  # fuse 128*512*512

        e3 = self.Maxpool2(e2)  # 128*128*128
        e3 = self.Conv3(e3)  # 256*128*128
        t3 = self.Maxpool2_d(t2)  # 128*128*128
        t3 = self.Conv3_d(t3)  # 256*128*128
        e3 = e3 + t3  # fuse 256*256*256

        e4 = self.Maxpool3(e3)  # 256*64*64
        e4 = self.Conv4(e4)  # 512*64*64
        t4 = self.Maxpool3_d(t3)
        t4 = self.Conv4_d(t4)
        e4 = e4 + t4  # 512*128*128

        e5 = self.Maxpool4(e4)  # 512*32*32
        e5 = self.Conv5(e5)  # 1024*32*32
        t5 = self.Maxpool4_d(t4)
        t5 = self.Conv5_d(t5)
        e5 = e5 + t5

        return e1, t1, e2, t2, e3, t3, e4, t4, e5

class DualEncoder_CWF(nn.Module):
    '''
    CWF: channel-weight-based Fusion
    '''
    def __init__(self, in_ch=3, d_ch=1, mode="ToRGB"):
        super(DualEncoder_CWF, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.mode = mode
        # adaptive fusion blocks
        self.l32 = Linear_tensor32()
        self.l64 = Linear_tensor64()
        self.l128 = Linear_tensor128()
        self.l256 = Linear_tensor256()
        self.l512 = Linear_tensor512()
        # rgb encoder
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        # depth
        self.Maxpool1_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5_d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_d = conv_block_d(d_ch, filters[0])
        self.Conv2_d = conv_block_d(filters[0], filters[1])
        self.Conv3_d = conv_block_d(filters[1], filters[2])
        self.Conv4_d = conv_block_d(filters[2], filters[3])
        self.Conv5_d = conv_block_d(filters[3], filters[4])

    def forward(self, x, y):
        if self.mode == "Bi-direction":
            e1 = self.Conv1(x)  # batchsize*64*512*512
            t1 = self.Conv1_d(y)  # 64*512*512
            et1 = self.l32(e1, t1)  # fuse 64*512*512
            e1 = e1 + et1
            t1 = t1 + et1

            e2 = self.Maxpool1(e1)  # 64*256*256
            e2 = self.Conv2(e2)  # 128*256*256
            t2 = self.Maxpool1_d(t1)  # 64*256*256
            t2 = self.Conv2_d(t2)  # 128*256*256
            et2 = self.l64(e2, t2)
            e2 = e2 + et2
            t2 = t2 + et2

            e3 = self.Maxpool2(e2)  # 128*128*128
            e3 = self.Conv3(e3)  # 256*128*128
            t3 = self.Maxpool2_d(t2)  # 128*128*128
            t3 = self.Conv3_d(t3)  # 256*128*128
            et3 = self.l128(e3, t3)
            e3 = e3 + et3
            t3 = t3 + et3

            e4 = self.Maxpool3(e3)  # 256*64*64
            e4 = self.Conv4(e4)  # 512*64*64
            t4 = self.Maxpool3_d(t3)
            t4 = self.Conv4_d(t4)
            et4 = self.l256(e4, t4)
            e4 = e4 + et4
            t4 = t4 + et4

            e5 = self.Maxpool4(e4)  # 512*32*32
            e5 = self.Conv5(e5)  # 1024*32*32
            t5 = self.Maxpool4_d(t4)
            t5 = self.Conv5_d(t5)
            et5 = self.l512(e5, t5)

            return e1, t1, e2, t2, e3, t3, e4, t4, et5

        if self.mode == "ToRGB":
            e1 = self.Conv1(x)  # batchsize*64*512*512
            t1 = self.Conv1_d(y)  # 64*512*512
            e1 = e1 + self.l32(e1, t1)  # fuse 64*512*512

            e2 = self.Maxpool1(e1)  # 64*256*256
            e2 = self.Conv2(e2)  # 128*256*256
            t2 = self.Maxpool1_d(t1)  # 64*256*256
            t2 = self.Conv2_d(t2)  # 128*256*256
            e2 = e2 + self.l64(e2, t2)

            e3 = self.Maxpool2(e2)  # 128*128*128
            e3 = self.Conv3(e3)  # 256*128*128
            t3 = self.Maxpool2_d(t2)  # 128*128*128
            t3 = self.Conv3_d(t3)  # 256*128*128
            e3 = e3 + self.l128(e3, t3)

            e4 = self.Maxpool3(e3)  # 256*64*64
            e4 = self.Conv4(e4)  # 512*64*64
            t4 = self.Maxpool3_d(t3)
            t4 = self.Conv4_d(t4)
            e4 = e4 + self.l256(e4, t4)

            e5 = self.Maxpool4(e4)  # 512*32*32
            e5 = self.Conv5(e5)  # 1024*32*32
            t5 = self.Maxpool4_d(t4)
            t5 = self.Conv5_d(t5)
            e5 = e5 + self.l512(e5, t5)

            return e1, t1, e2, t2, e3, t3, e4, t4, e5

        if self.mode == "ToDepth":
            e1 = self.Conv1(x)  # batchsize*64*512*512
            t1 = self.Conv1_d(y)  # 64*512*512
            t1 = t1 + self.l32(e1, t1)  # fuse 64*512*512

            e2 = self.Maxpool1(e1)  # 64*256*256
            e2 = self.Conv2(e2)  # 128*256*256
            t2 = self.Maxpool1_d(t1)  # 64*256*256
            t2 = self.Conv2_d(t2)  # 128*256*256
            t2 = t2 + self.l64(e2, t2)

            e3 = self.Maxpool2(e2)  # 128*128*128
            e3 = self.Conv3(e3)  # 256*128*128
            t3 = self.Maxpool2_d(t2)  # 128*128*128
            t3 = self.Conv3_d(t3)  # 256*128*128
            t3 = t3 + self.l128(e3, t3)

            e4 = self.Maxpool3(e3)  # 256*64*64
            e4 = self.Conv4(e4)  # 512*64*64
            t4 = self.Maxpool3_d(t3)
            t4 = self.Conv4_d(t4)
            t4 = t4 + self.l256(e4, t4)

            e5 = self.Maxpool4(e4)  # 512*32*32
            e5 = self.Conv5(e5)  # 1024*32*32
            t5 = self.Maxpool4_d(t4)
            t5 = self.Conv5_d(t5)
            t5 = t5 + self.l512(e5, t5)

            return e1, t1, e2, t2, e3, t3, e4, t4, t5

class DualEncoder_CWF_Number(nn.Module):
    '''
    CWF: channel-weight-based Fusion
    '''
    def __init__(self, in_ch=3, d_ch=1, number=4):
        super(DualEncoder_CWF_Number, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.number = number
        # adaptive fusion blocks
        self.l32 = Linear_tensor32()
        self.l64 = Linear_tensor64()
        self.l128 = Linear_tensor128()
        self.l256 = Linear_tensor256()
        self.l512 = Linear_tensor512()
        # rgb encoder
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        # depth
        self.Maxpool1_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5_d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_d = conv_block_d(d_ch, filters[0])
        self.Conv2_d = conv_block_d(filters[0], filters[1])
        self.Conv3_d = conv_block_d(filters[1], filters[2])
        self.Conv4_d = conv_block_d(filters[2], filters[3])
        self.Conv5_d = conv_block_d(filters[3], filters[4])

    def forward(self, x, y):
        if self.number == 4:
            e1 = self.Conv1(x)  # batchsize*64*512*512
            t1 = self.Conv1_d(y)  # 64*512*512
            e1 = e1 + self.l32(e1, t1)  # fuse 64*512*512

            e2 = self.Maxpool1(e1)  # 64*256*256
            e2 = self.Conv2(e2)  # 128*256*256
            t2 = self.Maxpool1_d(t1)  # 64*256*256
            t2 = self.Conv2_d(t2)  # 128*256*256
            e2 = e2 + self.l64(e2, t2)

            e3 = self.Maxpool2(e2)  # 128*128*128
            e3 = self.Conv3(e3)  # 256*128*128
            t3 = self.Maxpool2_d(t2)  # 128*128*128
            t3 = self.Conv3_d(t3)  # 256*128*128
            e3 = e3 + self.l128(e3, t3)

            e4 = self.Maxpool3(e3)  # 256*64*64
            e4 = self.Conv4(e4)  # 512*64*64
            t4 = self.Maxpool3_d(t3)
            t4 = self.Conv4_d(t4)
            e4 = e4 + self.l256(e4, t4)

            e5 = self.Maxpool4(e4)  # 512*32*32
            e5 = self.Conv5(e5)  # 1024*32*32
            t5 = self.Maxpool4_d(t4)
            t5 = self.Conv5_d(t5)
            e5 = e5 + t5

            return e1, t1, e2, t2, e3, t3, e4, t4, e5

        if self.number == 3:
            e1 = self.Conv1(x)  # batchsize*64*512*512
            t1 = self.Conv1_d(y)  # 64*512*512
            e1 = e1 + self.l32(e1, t1)  # fuse 64*512*512

            e2 = self.Maxpool1(e1)  # 64*256*256
            e2 = self.Conv2(e2)  # 128*256*256
            t2 = self.Maxpool1_d(t1)  # 64*256*256
            t2 = self.Conv2_d(t2)  # 128*256*256
            e2 = e2 + self.l64(e2, t2)

            e3 = self.Maxpool2(e2)  # 128*128*128
            e3 = self.Conv3(e3)  # 256*128*128
            t3 = self.Maxpool2_d(t2)  # 128*128*128
            t3 = self.Conv3_d(t3)  # 256*128*128
            e3 = e3 + self.l128(e3, t3)

            e4 = self.Maxpool3(e3)  # 256*64*64
            e4 = self.Conv4(e4)  # 512*64*64
            t4 = self.Maxpool3_d(t3)
            t4 = self.Conv4_d(t4)

            e5 = self.Maxpool4(e4)  # 512*32*32
            e5 = self.Conv5(e5)  # 1024*32*32
            t5 = self.Maxpool4_d(t4)
            t5 = self.Conv5_d(t5)
            e5 = e5 + t5

            return e1, t1, e2, t2, e3, t3, e4, t4, e5

        if self.number == 2:
            e1 = self.Conv1(x)  # batchsize*64*512*512
            t1 = self.Conv1_d(y)  # 64*512*512
            e1 = e1 + self.l32(e1, t1)  # fuse 64*512*512

            e2 = self.Maxpool1(e1)  # 64*256*256
            e2 = self.Conv2(e2)  # 128*256*256
            t2 = self.Maxpool1_d(t1)  # 64*256*256
            t2 = self.Conv2_d(t2)  # 128*256*256
            e2 = e2 + self.l64(e2, t2)

            e3 = self.Maxpool2(e2)  # 128*128*128
            e3 = self.Conv3(e3)  # 256*128*128
            t3 = self.Maxpool2_d(t2)  # 128*128*128
            t3 = self.Conv3_d(t3)  # 256*128*128

            e4 = self.Maxpool3(e3)  # 256*64*64
            e4 = self.Conv4(e4)  # 512*64*64
            t4 = self.Maxpool3_d(t3)
            t4 = self.Conv4_d(t4)

            e5 = self.Maxpool4(e4)  # 512*32*32
            e5 = self.Conv5(e5)  # 1024*32*32
            t5 = self.Maxpool4_d(t4)
            t5 = self.Conv5_d(t5)
            e5 = e5 + t5

            return e1, t1, e2, t2, e3, t3, e4, t4, e5

        if self.number == 1:
            e1 = self.Conv1(x)  # batchsize*64*512*512
            t1 = self.Conv1_d(y)  # 64*512*512
            e1 = e1 + self.l32(e1, t1)  # fuse 64*512*512

            e2 = self.Maxpool1(e1)  # 64*256*256
            e2 = self.Conv2(e2)  # 128*256*256
            t2 = self.Maxpool1_d(t1)  # 64*256*256
            t2 = self.Conv2_d(t2)  # 128*256*256

            e3 = self.Maxpool2(e2)  # 128*128*128
            e3 = self.Conv3(e3)  # 256*128*128
            t3 = self.Maxpool2_d(t2)  # 128*128*128
            t3 = self.Conv3_d(t3)  # 256*128*128

            e4 = self.Maxpool3(e3)  # 256*64*64
            e4 = self.Conv4(e4)  # 512*64*64
            t4 = self.Maxpool3_d(t3)
            t4 = self.Conv4_d(t4)

            e5 = self.Maxpool4(e4)  # 512*32*32
            e5 = self.Conv5(e5)  # 1024*32*32
            t5 = self.Maxpool4_d(t4)
            t5 = self.Conv5_d(t5)
            e5 = e5 + t5

            return e1, t1, e2, t2, e3, t3, e4, t4, e5

class DualEncoder_Add(nn.Module):
    def __init__(self, in_ch=3, d_ch=1):
        super(DualEncoder_Add, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # rgb encoder
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        # depth encoder
        self.Maxpool1_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4_d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5_d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_d = conv_block_d(d_ch, filters[0])
        self.Conv2_d = conv_block_d(filters[0], filters[1])
        self.Conv3_d = conv_block_d(filters[1], filters[2])
        self.Conv4_d = conv_block_d(filters[2], filters[3])
        self.Conv5_d = conv_block_d(filters[3], filters[4])

    def forward(self, x, y):
        e1 = self.Conv1(x)  # batchsize*64*512*512
        t1 = self.Conv1_d(y)  # 64*512*512
        e1 = e1 + t1  # fuse 64*512*512

        e2 = self.Maxpool1(e1)  # 64*256*256
        e2 = self.Conv2(e2)  # 128*256*256
        t2 = self.Maxpool1_d(t1)  # 64*256*256
        t2 = self.Conv2_d(t2)  # 128*256*256
        e2 = e2 + t2

        e3 = self.Maxpool2(e2)  # 128*128*128
        e3 = self.Conv3(e3)  # 256*128*128
        t3 = self.Maxpool2_d(t2)  # 128*128*128
        t3 = self.Conv3_d(t3)  # 256*128*128
        e3 = e3 + t3

        e4 = self.Maxpool3(e3)  # 256*64*64
        e4 = self.Conv4(e4)  # 512*64*64
        t4 = self.Maxpool3_d(t3)
        t4 = self.Conv4_d(t4)
        e4 = e4 + t4

        e5 = self.Maxpool4(e4)  # 512*32*32
        e5 = self.Conv5(e5)  # 1024*32*32
        t5 = self.Maxpool4_d(t4)
        t5 = self.Conv5_d(t5)
        e5 = e5 + t5

        return e1, t1, e2, t2, e3, t3, e4, t4, e5

###############Decoder#############
class single_Decoder(nn.Module):
    def __init__(self, out_ch=1):
        super(single_Decoder, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, e5, e4, t4, e3, t3, e2, t2, e1, t1):
        d5 = self.Up5(e5)  # 512*64*64
        d5 = torch.cat((e4, d5), dim=1)  # 1024*64*64
        d5 = self.Up_conv5(d5)  # 512*64*64

        d4 = self.Up4(d5)  # 256*128*128
        d4 = torch.cat((e3, d4), dim=1)  # 512*128*128
        d4 = self.Up_conv4(d4)  # 256*128*128

        d3 = self.Up3(d4)  # 128*256*256
        d3 = torch.cat((e2, d3), dim=1)  # 256*256*256
        d3 = self.Up_conv3(d3)  # 128*256*256

        d2 = self.Up2(d3)  # 64*512*512
        d2 = torch.cat((e1, d2), dim=1)  # 128*512*512
        d2 = self.Up_conv2(d2)  # 64*512*512

        out = self.Conv(d2)  # 5*512*512
        return out


class Decoder(nn.Module):
    def __init__(self, out_ch=1):
        super(Decoder, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, e5, e4, t4, e3, t3, e2, t2, e1, t1):
        d5 = self.Up5(e5)  # 512*64*64
        et4 = e4 + t4
        d5 = torch.cat((et4, d5), dim=1)  # 1024*64*64
        d5 = self.Up_conv5(d5)  # 512*64*64

        d4 = self.Up4(d5)  # 256*128*128
        et3 = e3 + t3
        d4 = torch.cat((et3, d4), dim=1)  # 512*128*128
        d4 = self.Up_conv4(d4)  # 256*128*128

        d3 = self.Up3(d4)  # 128*256*256
        et2 = e2 + t2
        d3 = torch.cat((et2, d3), dim=1)  # 256*256*256
        d3 = self.Up_conv3(d3)  # 128*256*256

        d2 = self.Up2(d3)  # 64*512*512
        et1 = e1 + t1
        d2 = torch.cat((et1, d2), dim=1)  # 128*512*512
        d2 = self.Up_conv2(d2)  # 64*512*512

        out = self.Conv(d2)  # 5*512*512
        return out

class Decoder_MSA(nn.Module):
    def __init__(self, out_ch=1):
        super(Decoder_MSA, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.msa_5 = MSA(in_ch=filters[3])
        self.msa_4 = MSA(in_ch=filters[2])
        self.msa_3 = MSA(in_ch=filters[1])
        self.msa_2 = MSA(in_ch=filters[0])

    def forward(self, e5, e4, t4, e3, t3, e2, t2, e1, t1):
        d5 = self.Up5(e5)  # 512*64*64
        mid_d5 = self.msa_5(e4, t4, d5)
        new_d5 = torch.cat((d5, mid_d5), dim=1)  # 1024*64*64
        new_d5 = self.Up_conv5(new_d5)  # 512*64*64

        d4 = self.Up4(new_d5)  # 256*128*128
        mid_d4 = self.msa_4(e3, t3, d4)
        new_d4 = torch.cat((d4, mid_d4), dim=1)  # 512*128*128
        new_d4 = self.Up_conv4(new_d4)  # 256*128*128

        d3 = self.Up3(new_d4)  # 128*256*256
        mid_d3 = self.msa_3(e2, t2, d3)
        new_d3 = torch.cat((d3, mid_d3), dim=1)  # 256*256*256
        new_d3 = self.Up_conv3(new_d3)  # 128*256*256

        d2 = self.Up2(new_d3)  # 64*512*512
        mid_d2 = self.msa_2(e1, t1, d2)
        new_d2 = torch.cat((d2, mid_d2), dim=1)  # 128*512*512
        new_d2 = self.Up_conv2(new_d2)  # 64*512*512
        out = self.Conv(new_d2)  # 5*512*512
        return out

class Decoder_MSA_filter(nn.Module):
    def __init__(self, out_ch=1):
        super(Decoder_MSA_filter, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.msa_5 = MSA_filter(in_ch=filters[3])
        self.conv5 = nn.Conv2d(filters[4], filters[3], kernel_size=1, bias=False)
        self.msa_4 = MSA_filter(in_ch=filters[2])
        self.conv4 = nn.Conv2d(filters[3], filters[2], kernel_size=1, bias=False)
        self.msa_3 = MSA_filter(in_ch=filters[1])
        self.conv3 = nn.Conv2d(filters[2], filters[1], kernel_size=1, bias=False)
        self.msa_2 = MSA_filter(in_ch=filters[0])
        self.conv2 = nn.Conv2d(filters[1], filters[0], kernel_size=1, bias=False)

    def forward(self, e5, e4, t4, e3, t3, e2, t2, e1, t1):
        d5 = self.Up5(e5)  # 512*64*64
        mid_d5 = self.conv5(self.msa_5(e4, t4))
        new_d5 = torch.cat((d5, mid_d5), dim=1)  # 1024*64*64
        new_d5 = self.Up_conv5(new_d5)  # 512*64*64

        d4 = self.Up4(new_d5)  # 256*128*128
        mid_d4 = self.conv4(self.msa_4(e3, t3))
        new_d4 = torch.cat((d4, mid_d4), dim=1)  # 512*128*128
        new_d4 = self.Up_conv4(new_d4)  # 256*128*128

        d3 = self.Up3(new_d4)  # 128*256*256
        mid_d3 = self.conv3(self.msa_3(e2, t2))
        new_d3 = torch.cat((d3, mid_d3), dim=1)  # 256*256*256
        new_d3 = self.Up_conv3(new_d3)  # 128*256*256

        d2 = self.Up2(new_d3)  # 64*512*512
        mid_d2 = self.conv2(self.msa_2(e1, t1))
        new_d2 = torch.cat((d2, mid_d2), dim=1)  # 128*512*512
        new_d2 = self.Up_conv2(new_d2)  # 64*512*512
        out = self.Conv(new_d2)  # 5*512*512
        return out

class Decoder_MSA_HL(nn.Module):
    def __init__(self, out_ch=5):
        super(Decoder_MSA_HL, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.msa_5 = MSA_HL(in_ch=filters[3])
        self.msa_4 = MSA_HL(in_ch=filters[2])
        self.msa_3 = MSA_HL(in_ch=filters[1])
        self.msa_2 = MSA_HL(in_ch=filters[0])

    def forward(self, e5, e4, t4, e3, t3, e2, t2, e1, t1):
        d5 = self.Up5(e5)  # 512*64*64
        mid_d5 = self.msa_5(torch.cat((e4, t4), dim=1), d5)
        new_d5 = torch.cat((d5, mid_d5), dim=1)  # 1024*64*64
        new_d5 = self.Up_conv5(new_d5)  # 512*64*64

        d4 = self.Up4(new_d5)  # 256*128*128
        mid_d4 = self.msa_4(torch.cat((e3, t3), dim=1), d4)
        new_d4 = torch.cat((d4, mid_d4), dim=1)  # 512*128*128
        new_d4 = self.Up_conv4(new_d4)  # 256*128*128

        d3 = self.Up3(new_d4)  # 128*256*256
        mid_d3 = self.msa_3(torch.cat((e2, t2), dim=1), d3)
        new_d3 = torch.cat((d3, mid_d3), dim=1)  # 256*256*256
        new_d3 = self.Up_conv3(new_d3)  # 128*256*256

        d2 = self.Up2(new_d3)  # 64*512*512
        mid_d2 = self.msa_2(torch.cat((e1, t1), dim=1), d2)
        new_d2 = torch.cat((d2, mid_d2), dim=1)  # 128*512*512
        new_d2 = self.Up_conv2(new_d2)  # 64*512*512
        out = self.Conv(new_d2)  # 5*512*512
        return out
