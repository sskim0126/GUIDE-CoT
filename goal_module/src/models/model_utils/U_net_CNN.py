import torch
import torch.nn as nn
import torchvision.transforms as TF
import clip
from torchvision.models import resnet50, ResNet50_Weights


class DoubleConv(nn.Module):
    """U-net double convolution block: (CNN => ReLU => BN) * 2"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_batch_norm=False,
                 ):
        super().__init__()
        block = []
        block.append(nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))

        block.append(nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class Encoder(nn.Module):
    """U-net encoder"""
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [DoubleConv(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class UpConv(nn.Module):
    """U-net Up-Conv layer. Can be real Up-Conv or bilinear up-sampling"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 up_mode='bilinear',
                 ):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2, stride=2, padding=0)
        elif up_mode == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2,
                            align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=1, padding=1))
        else:
            raise ValueError("No such up_mode")

    def forward(self, x):
        return self.up(x)


class Decoder(nn.Module):
    """U-net decoder, made of up-convolutions and CNN blocks.
    The cropping is necessary when 0-padding, due to the loss of
    border pixels in every convolution"""
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [UpConv(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList(
            [DoubleConv(2*chs[i + 1], chs[i + 1]) for i in range(len(chs) - 1)])

    def center_crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = TF.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            try:
                enc_ftrs = encoder_features[i]
                x = torch.cat([x, enc_ftrs], dim=1)
            except RuntimeError:
                enc_ftrs = self.center_crop(encoder_features[i], x)
                x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x


class DoubleConditionDecoder(nn.Module):
    """U-net decoder, made of up-convolutions and CNN blocks.
    The cropping is necessary when 0-padding, due to the loss of
    border pixels in every convolution"""
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [UpConv(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList(
            [DoubleConv(3*chs[i + 1], chs[i + 1]) for i in range(len(chs) - 1)])

    def center_crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = TF.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

    def forward(self, x, encoder_features, visual_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            try:
                enc_ftrs = encoder_features[i]
                vis_ftrs = visual_features[i]
                x = torch.cat([x, enc_ftrs, vis_ftrs], dim=1)
            except RuntimeError:
                enc_ftrs = self.center_crop(encoder_features[i], x)
                vis_ftrs = self.center_crop(visual_features[i], x)
                x = torch.cat([x, enc_ftrs, vis_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x


class OutputLayer(nn.Module):
    """U-net output layer: (CNN 1x1 => Sigmoid)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_layer = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=1, padding=0)
        # TODO: do not use sigmoid if you use BCEWithLogitsLoss
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.out_layer(x)


class VisSemUNet(nn.Module):
    """U-net architecture.
    retain_dim interpolate the output to have the same
    dimension of the input image"""
    def __init__(self,
                 enc_chs=(3, 64, 128, 256, 512, 1024),
                 dec_chs=(1024, 512, 256, 128, 64),
                 out_chs=1,
                 retain_dim=True,
                 clip_model="RN50",
                 ):
        super().__init__()
        model, preprocess = clip.load(clip_model)
        model = model.float()
        self.preprocess = TF.Compose([
            TF.ConvertImageDtype(torch.float32),
            preprocess.transforms[4],
        ])
        self.visual_encoder = model.visual
        bound_method = forward_prepool.__get__(
                self.visual_encoder, self.visual_encoder.__class__
            )
        setattr(self.visual_encoder, "forward", bound_method)
        self.encoder = Encoder(enc_chs)
        self.decoder = DoubleConditionDecoder(dec_chs)
        self.head = OutputLayer(dec_chs[-1], out_chs)
        self.retain_dim = retain_dim
        
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        
        self.clip_dims = (2048, 1024, 512, 256, 64, 64, 32, 32)
        self.conv_1x1 = nn.ModuleList()
        self.num_layers = len(dec_chs)
        for i in range(self.num_layers):
            self.conv_1x1.append(nn.Conv2d(self.clip_dims[i], dec_chs[i], kernel_size=1))

    def forward(self, x, rgb_image):
        _, _, H, W = x.shape
        
        rgb_image = self.preprocess(rgb_image)
        _, vis_ftrs = self.visual_encoder(rgb_image)
        vis_ftrs = vis_ftrs[::-1][:self.num_layers]
        enc_ftrs = self.encoder(x)
        enc_ftrs = enc_ftrs[::-1]
        
        for i in range(self.num_layers):
            vis_ftrs[i] = self.conv_1x1[i](vis_ftrs[i])
        
        out = self.decoder(enc_ftrs[0], enc_ftrs[1:], vis_ftrs[1:])
        out = self.head(out)
        if self.retain_dim:
            out = nn.functional.interpolate(out, (H, W))
        return out

def forward_prepool(self, x):
    """
    Adapted from https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L138
    Expects a batch of images where the batch number is even. The whole batch
    is passed through all layers except the last layer; the first half of the
    batch will be passed through avgpool and the second half will be passed
    through attnpool. The outputs of both pools are concatenated returned.
    """

    im_feats = []
    def stem(x):
        for conv, bn, relu in [(self.conv1, self.bn1, self.relu1), (self.conv2, self.bn2, self.relu2), (self.conv3, self.bn3, self.relu3)]:
            x = relu(bn(conv(x)))
            im_feats.append(x)
        x = self.avgpool(x)
        im_feats.append(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)

    for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
        x = layer(x)
        im_feats.append(x)
    return x, im_feats
