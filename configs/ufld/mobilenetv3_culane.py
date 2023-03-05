model = dict(
    name='Detector',
)

backbone = dict(
    name='MobileNetWrapper',
    net = 'MobileNetV3_large_x1_0_ssld',
    pretrain = True,
    out_conv=True,
)

featuremap_out_channel = 512
use_amp = True

griding_num = 200
num_classes = 4
heads = dict(name='LaneCls',
        dim = (griding_num + 1, 18, num_classes),
        loss = dict(name = 'SoftmaxFocalLoss',
                    gamma = 2),
        relation_loss = dict(name='ParsingRelationLoss'))

epochs = 51
batch_size = 30
total_iter = (88880 // batch_size + 1) * epochs 

lr_scheduler = dict(
    name = 'PolynomialDecay',
    learning_rate = 0.0003,
    decay_steps = total_iter
,
    power = 0.9
)

# optimizer = dict(
#   name = 'Momentum',
#   weight_decay = 1e-4,
#   momentum = 0.9
# )

optimizer = dict(
  name = 'AdamW',
)

ori_img_h = 590 
ori_img_w = 1640 
img_h = 288
img_w = 800
cut_height=0
sample_y = range(589, 230, -20)

img_norm = dict(
    mean=[0.408, 0.458, 0.485],
    std=[0.004, 0.004, 0.004]
)

row_anchor = 'culane_row_anchor'

train_process = [
    dict(name='RandomRotation', degree=(-6, 6)),
    dict(name='RandomUDoffsetLABEL', max_offset=100),
    dict(name='RandomLROffsetLABEL', max_offset=200),
    dict(name='GenerateLaneCls', row_anchor=row_anchor,
        num_cols=griding_num, num_classes=num_classes),
    dict(name='Resize', size=(img_w, img_h)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor', keys=['img', 'cls_label']),
]

val_process = [
    dict(name='Resize', size=(img_w, img_h)),
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

AMP = {'level': 'O1','save_dtype': 'float32','scale_loss': 1024,
        'auto_cast':{'level': 'O1','enable': True,'custom_white_list': {"elementwise_add", "batch_norm", "sync_batch_norm"},
                    'custom_black_list':{'bilinear_interp_v2'}}}

device = 'gpu'
seed =  0
save_inference_dir = './inference'
output_dir = './output_dir'
best_dir = './output_dir/best_dir'
pred_save_dir = './pred_save'
num_workers = 4
view = False
ignore_label = 255
use_visual = True # use visualdl to record training loss and accuracy
