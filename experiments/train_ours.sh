python train.py --data-path /data/marinedebris --model unet++ --batch-size 256 --project flobs-segm --no-label-refinement --no-marida --weight-decay 1e-12 --run-name unet++_2022-10-24_no-refinement_wde-12
python train.py --data-path /data/marinedebris --model unet++ --batch-size 256 --project flobs-segm --no-label-refinement --no-marida --weight-decay 1e-6 --run-name unet++_2022-10-24_no-refinement_no-marida

#python train.py --data-path /data/marinedebris --model unet++ --batch-size 256 --project flobs-segm --weight-decay 1e-6 --run-name unet++_2022-10-23
