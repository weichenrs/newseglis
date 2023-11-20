# Copyright (c) OpenMMLab. All rights reserved.
from .beit import BEiT
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .cgnet import CGNet
from .ddrnet import DDRNet
from .erfnet import ERFNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .icnet import ICNet
from .mae import MAE
from .mit import MixVisionTransformer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mscan import MSCAN
from .pidnet import PIDNet
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .stdc import STDCContextPathNet, STDCNet
from .swin import SwinTransformer
from .timm_backbone import TIMMBackbone
from .twins import PCPVT, SVT
from .unet import UNet
from .vit import VisionTransformer
from .vpd import VPD

from .sp_vit import SPVisionTransformer
from .my_vit import MYVisionTransformer
# from .sp_swin import SPSwinTransformer
# from .tp_swin import TPSwinTransformer
# from .vit_ds import VisionTransformer_ds
from .my_vit_fa import MYVisionTransformer_fa
from .my_vit_ds import MYVisionTransformer_ds
from .my_vit_fa_ds import MYVisionTransformer_fa_ds
from .my_vit_fa_nods import MYVisionTransformer_fa_nods

__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
    'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3',
    'VisionTransformer', 'SwinTransformer', 'MixVisionTransformer',
    'BiSeNetV1', 'BiSeNetV2', 'ICNet', 'TIMMBackbone', 'ERFNet', 'PCPVT',
    'SVT', 'STDCNet', 'STDCContextPathNet', 'BEiT', 'MAE', 'PIDNet', 'MSCAN',
    'DDRNet', 'VPD',
    'SPVisionTransformer', 'MYVisionTransformer',
    # 'VisionTransformer_ds', 
    'MYVisionTransformer_fa', 'MYVisionTransformer_ds',
    'MYVisionTransformer_fa_ds', 'MYVisionTransformer_fa_nods'

]
