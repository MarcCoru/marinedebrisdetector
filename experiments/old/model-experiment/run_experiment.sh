python train_segmentation.py --data-path /data/marinedebris --model manet --batch-size 128 --project flobs-segm-model --weight-decay 1e-8 --run-name manet
python train_segmentation.py --data-path /data/marinedebris --model unet --batch-size 128 --project flobs-segm-model --weight-decay 1e-8 --run-name unet
python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --project flobs-segm-model --weight-decay 1e-8 --run-name unet++
