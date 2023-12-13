# Copyright (c) OpenMMLab. All rights reserved.
from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .ddr_head import DDRHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .dpt_head import DPTHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .ham_head import LightHamHead
from .isa_head import ISAHead
from .knet_head import IterativeDecodeHead, KernelUpdateHead, KernelUpdator
from .lraspp_head import LRASPPHead
from .mask2former_head import Mask2FormerHead
from .maskformer_head import MaskFormerHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .pid_head import PIDHead
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .segformer_head import SegformerHead
from .segmenter_mask_head import SegmenterMaskTransformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .setr_mla_head import SETRMLAHead
from .setr_up_head import SETRUPHead
from .stdc_head import STDCHead
from .uper_head import UPerHead
from .vpd_depth_head import VPDDepthHead

# from .spatial_partition_decode_head import BaseSpatialPartitionDecodeHead
# from .sp_fcn_head import SPFCNHead
# from .sp_uper_head import SPUPerHead
from .uper_head_nopsp import UPerHead_nopsp
from .uper_head_bn import UPerHead_bn
from .vitsp_head import VITSPHead

from .fcn_head_sp import FCNHead_sp
from .uper_head_sp import UPerHead_sp
from .fcn_head_sp_comm1 import FCNHead_sp_comm1
from .uper_head_sp_comm1 import UPerHead_sp_comm1

from .atm_head import ATMHead
__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'PointHead', 'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead',
    'SETRMLAHead', 'DPTHead', 'SETRMLAHead', 'SegmenterMaskTransformerHead',
    'SegformerHead', 'ISAHead', 'STDCHead', 'IterativeDecodeHead',
    'KernelUpdateHead', 'KernelUpdator', 'MaskFormerHead', 'Mask2FormerHead',
    'LightHamHead', 'PIDHead', 'DDRHead', 'VPDDepthHead',
    # 'BaseSpatialPartitionDecodeHead', 'SPFCNHead', 'SPUPerHead'
    'UPerHead_nopsp', 'UPerHead_bn', 'VITSPHead',
    'FCNHead_sp', 'UPerHead_sp',    
    'FCNHead_sp_comm1', 'UPerHead_sp_comm1',
    'ATMHead'
]