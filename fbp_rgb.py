dataset_type = 'FBPDataset'
data_root = '../../../data/Five-Billion-Pixels/'
img_norm_cfg = dict(
    # mean=[126.607, 94.45, 99.45],  std=[61.50, 58.573, 56.635], to_rgb=True) #gid15_wrong
    # mean=[126.066, 93.966, 98.993, 91.906],  std=[65.501, 61.913, 59.451, 59.522], to_rgb=False) #fbp
    mean=[93.966, 98.993, 91.906],  std=[61.913, 59.451, 59.522], to_rgb=True) #fbp
    # [93.96666667  98.99333333  91.90666667]  [] 61.91381106 59.45110035 59.52254475] #gid15_rgb
    # mean = [103.601, 96.181, 71.313], std = [37.471, 29.267, 26.880], to_rgb=True) #deepglobe
    # [71.9356 81.7782 71.3334] [47.36788786 48.12360128 47.28967329] #cityscape
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) # cityscapes_imagenet
crop_size = (2048, 2048)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 2048), ratio_range=(1., 1.)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=(90, 270)),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend = 'tifffile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(6800, 7200),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='ori/Image_RGB/train',
        ann_dir='ori/Annotation__index/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='ori/Image_RGB/val',
        ann_dir='ori/Annotation__index/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='ori/Image_RGB/test',
        ann_dir='ori/Annotation__index/test',
        pipeline=test_pipeline))