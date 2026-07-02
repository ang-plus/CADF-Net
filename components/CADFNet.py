
import  torch
import  torch.nn as nn
from    ops.CDSGroups import CDSGroups
from    ops.pixelshuffle import pixelshuffle_block
import  torch.nn.functional as F

        
class CADFNet(nn.Module): 
    def __init__(self,num_in_ch=3,num_out_ch=3,num_feat=64,**kwargs):
        super(CADFNet, self).__init__()
        res_num     = kwargs["res_num"]
        up_scale    = kwargs["upsampling"]
        bias        = kwargs["bias"]
        residual_layer  = []
        self.res_num    = res_num
        for _ in range(res_num):
            temp_res = CDSGroups(channel_num=num_feat,**kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input  = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up     = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)
        self.window_size   = kwargs["window_size"]
        self.up_scale = up_scale

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        residual= self.input(x)
        out     = self.residual_layer(residual)
        out     = torch.add(self.output(out),residual)
        out     = self.up(out)
        out = out[:, :, :H*self.up_scale, :W*self.up_scale]
        return  out