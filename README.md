# TODO : Object Detection

- [x] Explain data augmentation​
- [x] Draw anchors​ 
- [x] Draw architecture​ 
- [x] Training targets​ : how are targets returned and converted when calculating loss? 
- [x] Qualitative evaluation​ at []()
- [x] Train a better detector ​at [utils/loss.py]() and [data/hyp/hyp.new_params.yaml]()


# Model Zoo
|Method|mAP_0.5|mAP_0.5:0.95|
|------|---|---|
|Baseline (Data Augment.)|0.306|0.144|
|No Data Augment.|0.188|0.083|
|No autoanchor|0.309|0.145|
|Focal Loss (obj)|||
|Focal Loss (cls)|||
|Focal Loss (obj+cls)|N/A|N/A|

The notation "N/A" indicates that the model does not need to be trained until the total epoch

because its performance was not sufficiently good initially.

## Train
#### Initial Trainig : Baseline Model with Data Augmentation 

```bash
CUDA_VISIBLE_DEVICES=2 python train_simple.py \
    --img 416 \
    --batch-size 256 \
    --epochs 80 \
    --data data/nuscenes.yaml \
    --weights yolov5n.pt \
    --workers 16 \
    --name yolov5n_yes_DA \
    --hyp data/hyps/hyp.pa2.yaml \
    --project aue8088-pa2 \
    --entity ophd
```

#### Ablation Study 1 : Effectiveness of Data Augmentation
```bash
python train_simple.py \
    --img 416 \
    --batch-size 256 \
    --epochs 80 \
    --data data/nuscenes.yaml \
    --weights yolov5n.pt \
    --workers 16 \
    --hyp data/hyps/hyp.no-augmentation.yaml \
    --name yolov5n_no_DA \
    --project aue8088-pa2 \
    --entity ophd
```

#### Ablation Study 2 : No Autoanchor
```bash
python train.py \
    --cfg models/yolov5n_nuscenes.yaml \
    --img 416 \
    --batch-size 256 \
    --epochs 80 \
    --noautoanchor \
    --data data/nuscenes.yaml \
    --weights yolov5n.pt \
    --workers 16 \
    --hyp data/hyps/hyp.pa2.yaml \
    --name yolov5n_noautoanchor_yes_DA \
    --project aue8088-pa2 \
    --entity ophd

```
#### Ablation Study 3-1 : Focal Loss for Obj and Clss  
```bash
python train_simple.py \
    --cfg models/yolov5n_nuscenes.yaml \
    --img 416 \
    --batch-size 256 \
    --epochs 80 \
    --data data/nuscenes.yaml \
    --weights yolov5n.pt \
    --workers 16 \
    --hyp data/hyps/hyp.focalloss_obj_cls.yaml \
    --name yolov5n_new_loss \
    --project aue8088-pa2 \
    --entity ophd

```
#### Ablation Study 3-2 : Focal Loss for Obj 
```bash
python train_simple.py \
    --cfg models/yolov5n_nuscenes.yaml \
    --img 416 \
    --batch-size 256 \
    --epochs 80 \
    --data data/nuscenes.yaml \
    --weights yolov5n.pt \
    --workers 16 \
    --hyp data/hyps/hyp.focalloss_obj.yaml \
    --name yolov5n_focalloss_obj \
    --project aue8088-pa2 \
    --entity ophd

```
#### Ablation Study 3-3 : Focal Loss for Clss  
```bash
python train_simple.py \
    --cfg models/yolov5n_nuscenes.yaml \
    --img 416 \
    --batch-size 256 \
    --epochs 80 \
    --data data/nuscenes.yaml \
    --weights yolov5n.pt \
    --workers 16 \
    --hyp data/hyps/hyp.focalloss_cls.yaml \
    --name yolov5n_focalloss_cls \
    --project aue8088-pa2 \
    --entity ophd
```

#### Ablation Study 4 : New Anchors 
```bash
python train_simple.py \
    --cfg models/yolov5n_nuscenes_more_anchors.yaml \
    --img 416 \
    --batch-size 256 \
    --epochs 80 \
    --data data/nuscenes.yaml \
    --weights yolov5n.pt \
    --workers 16 \
    --hyp data/hyps/hyp.pa2.yaml \
    --name yolov5n_new_anchors \
    --project aue8088-pa2 \
    --entity ophd
```


## Hyperparams Evolution [(ref : git issue)](https://github.com/ultralytics/yolov5/issues/607)
```bash
CUDA_VISIBLE_DEVICES=1 python train_simple.py \
    --img 415 \
    --batch-size 256 \
    --epochs 80 \
    --data data/nuscenes.yaml  \
    --weights yolov5n.pt \
    --workers 16 \
    --name yolov5_hyp_evol \
    --hyp data/hyps/hyp.new_params.yaml \
    --cache \
    --evolve 1000
```

After running this, you can get optimized hyper parameters

<details>
<summary>see evolve.yaml</summary>
<div markdown="1">

```yaml
evolve:
  metrics/precision: 0.33169
  metrics/recall: 0.16491
  metrics/mAP_0.5: 0.15665
  metrics/mAP_0.5:0.95: 0.063981
  val/box_loss: 0.089153
  val/obj_loss: 0.080776
  val/cls_loss: 0.048903
  lr0: 0.01
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1
  box: 0.05
  cls: 0.5
  cls_pw: 1
  obj: 1
  obj_pw: 1
  iou_t: 0.2
  anchor_t: 4
  fl_gamma: 0
  hsv_h: 0
  hsv_s: 0
  hsv_v: 0
  degrees: 0
  translate: 0
  scale: 0
  shear: 0
  perspective: 0
  flipud: 0
  fliplr: 0
  mosaic: 0
  mixup: 0
  copy_paste: 0
  anchors: 3
```

</div>
</details>



## Test
```bash
cd /path/to/datasets/nuscenes/images/
ffmpeg -framerate 24 -pattern_type glob -i '*.jpg' -c:v mpeg4 -b:v 10M -pix_fmt yuv420p output.mp4
```
You can generate a video with test images using command `ffmpeg`

And then, run inference like below 

#### Qualitative Evaluation for Image
```bash
CUDA_VISIBLE_DEVICES=1 python detect.py \
    --weights aue8088-pa2/yolov5n_yes_DA/weights/best.pt \
```

#### Qualitative Evaluation for Video
```bash
CUDA_VISIBLE_DEVICES=1 python detect.py \
    --weights aue8088-pa2/yolov5n_yes_DA/weights/best.pt \
    --source test_video.mp4 \
    --data data/nuscenes.yaml
    --source datasets/nuscenes/test/images/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295250162404.jpg \
    --data data/nuscenes.yaml
```

## Acknowledgement
you can check my work's log info at this [link](https://wandb.ai/ophd/aue8088-pa2?nw=nwuserophd) 
