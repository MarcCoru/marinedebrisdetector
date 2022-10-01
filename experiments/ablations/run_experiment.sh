python train_segmentation.py --data-path /scratch/izar/russwurm/marinedebris --model unet++ --batch-size 128
python train_segmentation.py --data-path /scratch/izar/russwurm/marinedebris --model unet++ --batch-size 128 --no-label-refinement
python train_segmentation.py --data-path /scratch/izar/russwurm/marinedebris --model unet++ --batch-size 128 --no-label-refinement --no-s2ships
python train_segmentation.py --data-path /scratch/izar/russwurm/marinedebris --model unet++ --batch-size 128 --no-s2ships
