
# TODO : Multispectral Pedestrian Detection
- [x] Train baseline model
- [x] Understand additional codes for this project
- [x] Update loss function at [`utils/loss.py`](https://github.com/MsDobby/AUE8088-PA2/blob/project/utils/loss.py#L168)
- [x] Adjust anchor box. See this [yaml file](https://github.com/MsDobby/AUE8088-PA2/blob/project/models/yolov5s_kaist-rgbt_more_anchor.yaml#L7)
- [x] Modify data augmenttion at [`utils/dataloaders.py`](https://github.com/MsDobby/AUE8088-PA2/blob/project/utils/dataloaders.py#L1219) and [`self.load_mosaic`](https://github.com/MsDobby/AUE8088-PA2/blob/project/utils/dataloaders.py#L909)
- [x] Plot miss rate @ FPPI
- [x] Further improvements 


# Prerequisite
### K-fold cross validation
```bash
cd datasetes
python gen_k_fold_annots.py \
    --k 5 \
    --idx 0 \
    --datasets kaist_rgbt
```

You can get three files 
- `train_{idx}_{k}.txt`
- `val_{idx}_{k}.txt`
- `annots_{idx}_{k}.json`

And these are for 5-fold cross validation at TODO 1 

## Train
#### Initial Trainig : Baseline Model

```bash
python train_simple.py \
  --img 640 \
  --batch-size 16 \
  --epochs 20 \
  --data data/kaist-rgbt.yaml \
  --cfg models/yolov5n_kaist-rgbt.yaml \
  --weights yolov5n.pt \
  --workers 16 \
  --name yolov5n-rgbt \
  --rgbt \
  --single-cls \
  --project aue8088-project \
  --entity ophd
```

#### Ablation Study 1 : Effectiveness of Central Loss for class "people"
```bash
python train_simple.py \
  --img 640   \
  --batch-size 16   \
  --epochs 20   \
  --data data/kaist-rgbt.yaml   \
  --cfg models/yolov5s_kaist-rgbt_more_anchor.yaml   \
  --weights yolov5n.pt   \
  --workers 16   \
  --name yolov5n-rgbt_centerloss_and_more_anchors   \
  --rgbt   \
  --project aue8088-project   \
  --entity ophd   \
  --device 3
```

#### Ablation Study 2 : Focal loss for object and class loss
```bash
python train_simple.py \
  --img 640 \
  --batch-size 16 \
  --epochs 20 \
  --data data/kaist-rgbt.yaml \
  --hyp data/hyps/hyp.focal_loss.yaml \
  --cfg models/yolov5s_kaist-rgbt_more_anchor.yaml \
  --weights yolov5n.pt \
  --workers 16 \
  --name yolov5n-rgbt_fcloss_anchors \
  --rgbt \
  --project aue8088-project \
  --entity ophd \
  --device 2
```

## Acknowledgement
you can check my work's log info at this [link](https://wandb.ai/ophd/aue8088-project?nw=nwuserophd) 
