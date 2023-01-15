model = dict(
    name='Detector',
)

backbone = dict(
    name='ResNetWrapper',
    resnet='resnet50',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
    in_channels=[64, 128, 256, 512]
)

sample_y = range(590, 270, -8)

batch_size = 24
aggregator = dict(
    name='TransConvEncoderModule',
    in_dim=2048,
    attn_in_dims=[2048, 256],
    attn_out_dims=[256, 256],
    strides=[1, 1],
    ratios=[4, 4],
    pos_shape=(batch_size, 10, 25),
)

neck=dict(
    name='FPN',
    in_channels=[256, 512, 1024, 256],
    out_channels=64,
    num_outs=4,
    #trans_idx=-1,
)

loss_weights=dict(
        hm_weight=1,
        kps_weight=0.4,
        row_weight=1.,
        range_weight=1.,
    )

num_lane_classes=1
heads=dict(
    name='CondLaneHead',
    heads=dict(hm=num_lane_classes),
    in_channels=(64, ),
    num_classes=num_lane_classes,
    head_channels=64,
    head_layers=1,
    disable_coords=False,
    branch_in_channels=64,
    branch_channels=64,
    branch_out_channels=64,
    reg_branch_channels=64,
    branch_num_conv=1,
    hm_idx=2,
    mask_idx=0,
    compute_locations_pre=True,
    location_configs=dict(size=(batch_size, 1, 80, 200), device='cuda:0'),
    crit_loss = dict(name = 'Condlane_focalLoss'),
    crit_kp_loss = dict(name = 'RegL1KpLoss'),
    crit_ce_loss = dict(name = 'CrossEntropyLoss_nn')
    
)

epochs = 16
total_iter = (88880 // batch_size) * epochs

lr_scheduler = dict(
    name = 'MultiStepDecay',
    learning_rate = 3e-4,
    milestones=[8,14], 
    gamma=0.1
)

optimizer = dict(name='AdamW',  
                 beta1 =  0.9,
                 beta2 = 0.999, 
                 epsilon=1e-8)

seg_loss_weight = 1.0
eval_ep = 1
save_ep = 1 

img_norm = dict(
    mean=[75.3, 76.6, 77.6],
    std=[50.5, 53.8, 54.3]
)

img_height = 320 
img_width = 800
cut_height = 0 
ori_img_h = 590
ori_img_w = 1640

mask_down_scale = 4
hm_down_scale = 16
num_lane_classes = 1
line_width = 3
radius = 6
nms_thr = 4
img_scale = (800, 320)
crop_bbox = [0, 270, 1640, 590]
mask_size = (1, 80, 200)

train_process = [
    dict(name='Alaug',
    transforms=[dict(name='Compose', params=dict(bboxes=False, keypoints=True, masks=False)),
    dict(
        name='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1),
    dict(name='Resize', height=img_scale[1], width=img_scale[0], p=1),
    dict(
        name='OneOf',
        transforms=[
            dict(
                name='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                name='HueSaturationValue',
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-15, 15),
                val_shift_limit=(-10, 10),
                p=1.0),
        ],
        p=0.7),
    dict(name='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(
        name='OneOf',
        transforms=[
            dict(name='Blur', blur_limit=3, p=1.0),
            dict(name='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.2),
    dict(name='RandomBrightness', limit=0.2, p=0.6),
    dict(
        name='ShiftScaleRotate',
        shift_limit=0.1,
        scale_limit=(-0.2, 0.2),
        rotate_limit=10,
        border_mode=0,
        p=0.6),
    dict(
        name='RandomResizedCrop',
        height=img_scale[1],
        width=img_scale[0],
        scale=(0.8, 1.2),
        ratio=(1.7, 2.7),
        p=0.6),
    dict(name='Resize', height=img_scale[1], width=img_scale[0], p=1),]
    ),
    dict(name='CollectLane',
        down_scale=mask_down_scale,
        hm_down_scale=hm_down_scale,
        max_mask_sample=5,
        line_width=line_width,
        radius=radius,
        keys=['img', 'gt_hm'],
        meta_keys=[
            'gt_masks', 'mask_shape', 'hm_shape',
            'down_scale', 'hm_down_scale', 'gt_points'
        ]
    ),
    #dict(name='Resize', size=(img_width, img_height)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor', keys=['img', 'gt_hm'], collect_keys=['img_metas']),
]


val_process = [
    dict(name='Alaug',
        transforms=[dict(name='Compose', params=dict(bboxes=False, keypoints=True, masks=False)),
            dict(name='Crop',
            x_min=crop_bbox[0],
            x_max=crop_bbox[2],
            y_min=crop_bbox[1],
            y_max=crop_bbox[3],
            p=1),
        dict(name='Resize', height=img_scale[1], width=img_scale[0], p=1)]
    ),
    #dict(name='Resize', size=(img_width, img_height)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor', keys=['img']),
]

dataset_path = '/home/aistudio/data'
dataset = dict(
    train=dict(
        name='CULane',
        data_root=dataset_path,
        split='train',
        processes=train_process,
    ),
    val=dict(
        name='CULane',
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        name='CULane',
        data_root=dataset_path,
        split='test',
        processes=val_process,
    )
)

log_config = dict(
    name = 'LogHook',
    interval = 50
    )

custom_config = [dict(
    name = 'EvaluateHook'
    )]

device = 'gpu'
seed =  0
save_inference_dir = './inference'
output_dir = './output_dir'
best_dir = './output_dir/best_dir'
pred_save_dir = './pred_save'
num_workers = 4
view = False
ignore_label = 255