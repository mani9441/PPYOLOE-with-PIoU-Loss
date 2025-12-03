_base_ = '/home/computador/Desktop/models_research/loss/mmyolo/configs/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py'


# Dataset root
data_root = '/home/computador/Desktop/models_research/mmyolo/Dataset/'
class_name = ('Seal', 'Tag_White', 'Tag_Yellow')
num_classes = len(class_name)
metainfo = dict(classes=class_name)


# Model config
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
    
        loss_bbox=dict(
            type='PIOULoss',
            iou_mode='piou',       # for compatibility with loss builder
            bbox_format='xyxy',    # PP-YOLOE uses xyxy internally
            reduction='mean',
            loss_weight=2.5,
            return_iou=False
        )
    ),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes)
    )
)


# 🔧 Dataloaders
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        # pipeline=[
        #     dict(type='LoadImageFromFile'),
        #     dict(type='LoadAnnotations', with_bbox=True),
        #     dict(type='YOLOv5CocoDataset', ann_file='train/_annotations.coco.json')
        # ]
    )
)

val_dataloader = dict(
    batch_size=6,
    num_workers=4,
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')
    )
)


test_dataloader = dict(
    batch_size=6,
    num_workers=4,
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/_annotations.coco.json',   
        data_prefix=dict(img='test/')       
    )
)

# Evaluators

val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = dict(ann_file=data_root + 'test/_annotations.coco.json')

# Training schedule
max_epochs = 100
train_cfg = dict(max_epochs=max_epochs, val_interval=10)

# Logger & checkpoints
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(
        warmup_min_iter=10,
        warmup_epochs=3,
        total_epochs=int(max_epochs * 1.2)
    )
)

# Load pretrained weights (still use PP-YOLOE-S pretrained)
load_from = 'https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco/ppyoloe_plus_s_fast_8xb8-80e_coco_20230101_154052-9fee7619.pth'

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend')]
)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=5e-4)
)
