
## Project. Multispectral Pedestrian Detection

### TODO 1 : K-fold cross validation

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

And these are 