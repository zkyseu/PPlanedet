model = dict(
    name='Detector',
)

#backbone = dict(
#    name='ResNet',
#    output_stride=8,
#    multi_grid=[1, 1, 1],
#    return_idx=[0,1,2],
#    pretrained='https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz',
#)

backbone = dict(
    name='ResNetWrapper',
    resnet='resnet50',
    pretrained=True,
    replace_stride_with_dilation=[False, True, False],
    out_conv=True,
    in_channels=[64, 128, 256, -1]
)

featuremap_out_channel = 128
featuremap_out_stride = 8
sample_y = range(589, 230, -20)

aggregator = dict(
    name='SCNN',
)

heads = dict( 
    name='LaneSeg',
    decoder=dict(name='PlainDecoder'),
    exist=dict(name='ExistHead'),
    thr=0.3,
    sample_y=sample_y,
    seg_loss = dict(name = 'CrossEntropyLoss',
                   weight = (0.4,1,1,1,1),
                   loss_weight = 1.),
    exist_loss = dict(name = 'BCELoss',
                      weight = 'dynamic',
                      loss_weight = 0.1)
)


epochs = 12
batch_size = 8
total_iter = (88880 // batch_size) * epochs

lr_scheduler = dict(
    name = 'PolynomialDecay',
    learning_rate = 0.005,
    decay_steps = total_iter
)

optimizer = dict(
  name = 'Momentum',
  weight_decay = 1e-4,
  momentum = 0.9
)

img_height = 288
img_width = 800
cut_height = 240 
ori_img_h = 590
ori_img_w = 1640

img_norm = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


train_process = [
    dict(name='RandomRotation', degree=(-2, 2)),
    dict(name='RandomHorizontalFlip'),
    dict(name='Resize', size=(img_width, img_height)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor', keys=['img', 'mask', 'lane_exist']),
]

val_process = [
    dict(name='Resize', size=(img_width, img_height)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor', keys=['img']),
]

dataset_path = '/home/fyj/zky/tusimple/culane'
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
num_classes = 4 + 1
exist_num_class = num_classes
view = False
ignore_label = 255






