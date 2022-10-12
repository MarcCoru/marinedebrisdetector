python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --no-label-refinement --project flobs-segm-hrbands --weight-decay 1e-8 --run-name flobs-segm-hrbands --hr-only
python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --no-label-refinement --project flobs-segm-hrbands --weight-decay 1e-8 --run-name flobs-segm-allbands
