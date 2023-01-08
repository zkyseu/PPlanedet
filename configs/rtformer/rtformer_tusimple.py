model = dict(
    name='Detector',
)

backbone = dict(
    name='RTFormer',
    num_classes = 6+1,
    base_channels = 64,
    head_channels = 128,
    use_injection = [True,False],
    use_aux_head = False,
    pretrained = 'https://paddleseg.bj.bcebos.com/dygraph/backbone/rtformer_base_backbone_imagenet_pretrained.zip'
)

sample_y=range(710, 150, -10)

heads = dict(
    name='ERF_head',
    thr=0.6,
    seg_loss = dict(name = 'CrossEntropyLoss',
                   weight = (0.4,1,1,1,1,1,1),
                   loss_weight = 1.),
    sample_y=sample_y,
    exist_loss = dict(name = 'BCELoss',
                      weight = 'dynamic',
                      loss_weight = 0.1)
)

epochs = 100
batch_size = 24
total_iter = (3616 // batch_size + 1) * epochs 

lr_scheduler = dict(
    name = 'PolynomialDecay',
    learning_rate = 0.0001,
    decay_steps = total_iter,
    end_lr = 0.000001
)

optimizer = dict(
 name = 'AdamW',
 beta1 =  0.9,
 beta2 = 0.999,
 weight_decay = 0.0125
)

img_height = 368
img_width = 640
cut_height = 160
ori_img_h = 720
ori_img_w = 1280

img_norm = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = [
    dict(name='RandomRotation'),
    dict(name='RandomHorizontalFlip'),
    dict(name='Resize', size=(img_width, img_height)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor',keys=['img', 'mask', 'lane_exist']),
] 

val_transform = [
    dict(name='Resize', size=(img_width, img_height)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor', keys=['img']),
] 

dataset_path = '/home/fyj/zky/tusimple'

dataset = dict(
    train=dict(
        name='TuSimple',
        data_root=dataset_path,
        split='trainval',
        processes=train_transform,
    ),
    val=dict(
        name='TuSimple',
        data_root=dataset_path,
        split='test',
        processes=val_transform,
    ),
    test=dict(
        name='TuSimple',
        data_root=dataset_path,
        split='test',
        processes=val_transform,
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
num_classes = 6 + 1
view = False
ignore_label = 255
test_json_file='/home/fyj/zky/tusimple/test_label.json'
