python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --no-label-refinement --project flobs-segm-weight-decay --weight-decay 1e-12 --run-name wde-12
python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --no-label-refinement --project flobs-segm-weight-decay --weight-decay 1e-8 --run-name wde-8
python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --no-label-refinement --project flobs-segm-weight-decay --weight-decay 1e-4 --run-name wde-4

python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --no-label-refinement --project flobs-segm-lr --learning-rate 1e-2 --run-name lre-2
python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --no-label-refinement --project flobs-segm-lr --learning-rate 1e-3 --run-name lre-3
python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --no-label-refinement --project flobs-segm-lr --learning-rate 1e-4 --run-name lre-4
