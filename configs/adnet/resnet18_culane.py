model = dict(name='Detector',)
# basic setting
img_w = 800
img_h = 320
# work_dir = './test_dir/culane/res18_r6_app'
# network setting
fpn_down_scale = [8,16,32]
anchors_num = 500
num_points = 72
max_lanes = 5

backbone = dict(
    name='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

neck = dict(name='SA_FPN',
            in_channels=[128,256,512],
            out_channels=64,
            num_outs=3)

heads = dict(name='SPGHead',
        img_width = img_w,
        img_height = img_h,
        start_points_num=anchors_num,
        hm_focalloss=dict(name='GFocalLoss'),
        theta_regloss=dict(name='RegL1KpLoss'),
        focal_loss=dict(name='FocalLoss_cls',alpha=0.25),
        liou_loss=dict(name='GLiou_loss'))

# train setting
regw = 6
hmw = 2
thetalossw = 3
cls_loss_w = 6
lr = 0.0008

epochs = 15
batch_size = 90
batch_size_test = 200

dynamic_after = 6
eval_ep = 3
save_ep = 1
do_mask = False

lr_scheduler = dict(name='CosineAnnealingDecay',learning_rate = 0.0016, T_max=epochs)
optimizer = dict(name='AdamW')  # 3e-4 for batchsize 8

train_parameters = dict(
    conf_threshold=None,
    nms_thres=0.8,
    nms_topk=max_lanes
)

test_parameters = dict(conf_threshold=0.3, nms_thres=0.8, nms_topk=max_lanes)

sample_y=range(589, 230, -1)
ori_img_w=1640
ori_img_h=590
cut_height=270
hm_down_scale = 8
keys = ['img', 'lane_line','gt_hm','shape_hm','shape_hm_mask']
train_process = [
    dict(
        name='GenerateLanePts',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],

    ),
    dict(name='CollectHm',
    down_scale=hm_down_scale,
    hm_down_scale=hm_down_scale,
    max_mask_sample=5,
    line_width=3,
    # 12
    radius=12,
    theta_thr = 0.5,
    keys=keys,
    meta_keys=['gt_points']
    ),    
    dict(name='ToTensor', keys=keys),
]

val_process = [
        dict(
        name='GenerateLanePts',
        training = False,
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],

    ),
    dict(name='ToTensor', keys=['img','lane_line']),
]

dataset_path = '/paddle/project/dataset'
dataset_type = 'CULane'
dataset = dict(
    train=dict(
        name=dataset_type,
        data_root=dataset_path,
        split='train',
        processes=train_process,
    ),
    val=dict(
        name=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        name=dataset_type,
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


num_workers = 4
device = 'gpu'
seed=0
lr_update_by_epoch = False
save_inference_dir = './inference'
output_dir = './output/'
best_dir = './output//best_dir'
pred_save_dir = './pred_save'
