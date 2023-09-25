model = dict(name='Detector', )

backbone = dict(
    name='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)
featuremap_out_channel = 192

heads = dict(name='DETRHead',
             cls_loss = dict(name = 'FocalLoss_cls',alpha=0.25),
             liou_loss = dict(name = 'Liou_loss'))
                   
iou_loss_weight = 2.
cls_loss_weight = 2.
xyt_loss_weight = 0.2
seg_loss_weight = 1.0

neck = dict(name='DETRTransformer',
            backbone_num_channels=512)

test_parameters = dict(conf_threshold=0.40, nms_topk=max_lanes)

epochs = 15
batch_size = 48 

lr_scheduler = dict(name='CosineAnnealingDecay',learning_rate = 0.6e-3, T_max=epochs)
optimizer = dict(name='AdamW')  # 3e-4 for batchsize 8

eval_ep = 1
save_ep = epochs

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1640
ori_img_h = 590
img_height = 320
img_width = 800
cut_height = 270 

train_process = [
    dict(
        name='GenerateCLRLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_height, width=img_width)),
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
                 parameters=dict(size=dict(height=img_height, width=img_width)),
                 p=1.0),
        ],
    ),
    dict(name='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(name='GenerateCLRLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_height, width=img_width)),
                  p=1.0),
         ],
         training=False),
    dict(name='ToTensor', keys=['img']),
]

dataset_path = '/root/paddlejob/workspace/train_data/datasets'
dataset_name = 'CULane'
dataset = dict(train=dict(
    name=dataset_name,
    data_root=dataset_path,
    split='train',
    processes=train_process,
),
val=dict(
    name=dataset_name,
    data_root=dataset_path,
    split='test',
    processes=val_process,
),
test=dict(
    name=dataset_name,
    data_root=dataset_path,
    split='test',
    processes=val_process,
))

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
num_classes = 5
view = False
ignore_label = 255
seg=False
