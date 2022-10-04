python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --no-label-refinement --project flobs-segm-labelrefinement --weight-decay 1e-8 --run-name flobs-segm-wo-refinement
python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --project flobs-segm-labelrefinement --weight-decay 1e-8 --run-name flobs-segm-with-refinement
