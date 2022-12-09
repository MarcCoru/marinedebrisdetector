for seed in 1 2 3
do

python train.py  \
  --data-path /data/marinedebris  \
  --model unet++  \
  --workers 32 \
  --batch-size 256  \
  --project marinedebrisdetector \
  --run-name unet++$seed \
  --seed $seed

done
