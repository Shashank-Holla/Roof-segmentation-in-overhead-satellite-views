import torch
import torch.nn as nn

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class ResnetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetEncoderBlock, self).__init__()
        self.double_conv = DoubleConvBlock(in_channels, out_channels)
        self.down = nn.MaxPool2d(2)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        identity = self.skip_conv(x)
        out = self.double_conv(x)
        out = out + identity
        return self.down(out)


class ResnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetDecoderBlock, self).__init__()
        # decoder output is of higher channel. Reducing it to output channel.
        self.transition_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # bypass double convolution. But after concat the shape is doubled. Hence, matching the skip connection's channel size. 
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.double_conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, encoder_input):
        x = self.transition_conv(x)
        x = self.up(x)
        x = torch.cat([x, encoder_input], dim=1)
        identity = self.skip_conv(x)
        out = self.double_conv(x)
        out = out + identity
        return out

class SatUMaskNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SatUMaskNet, self).__init__()
        # Input convolution
        self.input_conv = DoubleConvBlock(in_channels, 32)
        # Encoder blocks
        self.encode1 = ResnetEncoderBlock(32, 64)
        self.encode2 = ResnetEncoderBlock(64, 128)
        self.encode3 = ResnetEncoderBlock(128, 256)
        self.encode4 = ResnetEncoderBlock(256, 512)

        # Decoder blocks
        self.decode1 = ResnetDecoderBlock(512, 256)
        self.decode2 = ResnetDecoderBlock(256, 128)
        self.decode3 = ResnetDecoderBlock(128, 64)
        self.decode4 = ResnetDecoderBlock(64, 32)

        # output convolution
        self.output_conv = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        x = self.input_conv(x)
        # Encoder section
        enc1 = self.encode1(x)
        enc2 = self.encode2(enc1)
        enc3 = self.encode3(enc2)
        enc4 = self.encode4(enc3)
        # Decoder section
        dec1 = self.decode1(enc4, enc3)
        dec2 = self.decode2(dec1, enc2)
        dec3 = self.decode3(dec2, enc1)
        dec4 = self.decode4(dec3, x)
        # prepare output
        mask = self.output_conv(dec4)
        return mask