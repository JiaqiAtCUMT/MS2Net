from __future__ import print_function, division
import torch.nn as nn
from model.base_model import Encoder, Decoder, DualEncoder_CWF, Decoder_MSA, Decoder_MSA_HL, Decoder_MSA_filter, \
    DualEncoder_CWF_Number, DualEncoder_Add, DualEncoder, single_Decoder


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=5):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_ch=in_ch)
        self.decoder = Decoder(out_ch=out_ch)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        out = self.decoder(e5, e4, e3, e2, e1)

        return out

############################Proposed Method#############################(Done)
class DualUNet_CWF_MSA(nn.Module):
    def __init__(self, in_ch=3, d_ch=1, out_ch=5, mode="ToRGB"):
        super(DualUNet_CWF_MSA, self).__init__()
        self.encoder = DualEncoder_CWF(in_ch=in_ch, d_ch=d_ch, mode=mode)
        self.decoder = Decoder_MSA(out_ch=out_ch)

    def forward(self, x, y):
        e1, t1, e2, t2, e3, t3, e4, t4, e5 = self.encoder(x, y)
        out = self.decoder(e5, e4, t4, e3, t3, e2, t2, e1, t1)

        return out


#############Ablation Study of Multi-source Attention##############
class DualUNet_CWF_filter(nn.Module):
    def __init__(self, in_ch=3, d_ch=1, out_ch=5, mode="ToRGB"):
        super(DualUNet_CWF_filter, self).__init__()
        self.encoder = DualEncoder_CWF(in_ch=in_ch, d_ch=d_ch, mode=mode)
        self.decoder = Decoder_MSA_filter(out_ch=out_ch)

    def forward(self, x, y):
        e1, t1, e2, t2, e3, t3, e4, t4, e5 = self.encoder(x, y)
        out = self.decoder(e5, e4, t4, e3, t3, e2, t2, e1, t1)

        return out


class DualUNet_CWF_HL(nn.Module):
    def __init__(self, in_ch=3, d_ch=1, out_ch=5, mode="ToRGB"):
        super(DualUNet_CWF_HL, self).__init__()
        self.encoder = DualEncoder_CWF(in_ch=in_ch, d_ch=d_ch, mode=mode)
        self.decoder = Decoder_MSA_HL(out_ch=out_ch)

    def forward(self, x, y):
        e1, t1, e2, t2, e3, t3, e4, t4, e5 = self.encoder(x, y)
        out = self.decoder(e5, e4, t4, e3, t3, e2, t2, e1, t1)

        return out


################Ablation Study of the number of Channel-Weight-Based Fsuion####################(Done)
class DualUNet_CWF_MSA_Number(nn.Module):
    def __init__(self, in_ch=3, d_ch=1, out_ch=5, number=4):
        super(DualUNet_CWF_MSA_Number, self).__init__()
        self.encoder = DualEncoder_CWF_Number(in_ch, d_ch, number)
        self.decoder = Decoder_MSA(out_ch=out_ch)

    def forward(self, x, y):
        e1, t1, e2, t2, e3, t3, e4, t4, e5 = self.encoder(x, y)
        out = self.decoder(e5, e4, t4, e3, t3, e2, t2, e1, t1)

        return out

###############Ablation Study of simple addition and Channel-Weight-Based Fusion###################
class DualUNet_Add_MSA(nn.Module):
    def __init__(self, in_ch=3, d_ch=1, out_ch=5):
        super(DualUNet_Add_MSA, self).__init__()
        self.encoder = DualEncoder_Add(in_ch, d_ch)
        self.decoder = Decoder_MSA(out_ch=out_ch)

    def forward(self, x, y):
        e1, t1, e2, t2, e3, t3, e4, t4, e5 = self.encoder(x, y)
        out = self.decoder(e5, e4, t4, e3, t3, e2, t2, e1, t1)

        return out


################################### Ablation Study of CWF and MSA #################################
class DualUNet(nn.Module):
    def __init__(self, in_ch=3, d_ch=1, out_ch=5):
        super(DualUNet, self).__init__()
        self.encoder = DualEncoder(in_ch=in_ch, d_ch=d_ch)
        self.decoder = single_Decoder(out_ch=out_ch)
    def forward(self, x, y):
        e1, t1, e2, t2, e3, t3, e4, t4, e5 = self.encoder(x, y)
        out = self.decoder(e5, e4, t4, e3, t3, e2, t2, e1, t1)

        return out

class DualUNet_CWF(nn.Module):
    def __init__(self, in_ch=3, d_ch=1, out_ch=5, mode="ToRGB"):
        super(DualUNet_CWF, self).__init__()
        self.encoder = DualEncoder_CWF(in_ch=in_ch, d_ch=d_ch, mode=mode)
        self.decoder = Decoder(out_ch=out_ch)

    def forward(self, x, y):
        e1, t1, e2, t2, e3, t3, e4, t4, e5 = self.encoder(x, y)
        out = self.decoder(e5, e4, t4, e3, t3, e2, t2, e1, t1)
        return out

class DualUNet_MSA(nn.Module):
    def __init__(self, in_ch=3, d_ch=1, out_ch=5):
        super(DualUNet_MSA, self).__init__()
        self.encoder = DualEncoder(in_ch=in_ch, d_ch=d_ch)
        self.decoder = Decoder_MSA(out_ch=out_ch)

    def forward(self, x, y):
        e1, t1, e2, t2, e3, t3, e4, t4, e5 = self.encoder(x, y)
        out = self.decoder(e5, e4, t4, e3, t3, e2, t2, e1, t1)

        return out