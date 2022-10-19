python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 256 --project flobs-segm --weight-decay 1e-8 --run-name unet++_2022-10-20_fixed_marida

#python train_segmentation.py --data-path /data/marinedebris --model manet --batch-size 256 --project flobs-segm --weight-decay 1e-8 --run-name manet_2022-10-18_new_refinement

#python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 256 --project flobs-segm --weight-decay 1e-12 --run-name unet++_2022-10-18_new_refinement

#python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --no-label-refinement --project flobs-segm-labelrefinement --weight-decay 1e-8 --run-name flobs-segm-wo-refinement
#python train_segmentation.py --data-path /data/marinedebris --model unet++ --batch-size 128 --project flobs-segm-labelrefinement --weight-decay 1e-8 --run-name flobs-segm-with-refinement
