# Model Training

more details in a dedicated [model training page]()
see full training scripts in [training scripts](training_scripts) folder.
We use weights and biases for logging. Check the previous training runs [here](https://wandb.ai/marccoru/marinedebrisdetector)

## Detailed Setup 

### Python Environment
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```
python train.py  \
  --data-path /data/marinedebris  \
  --model unet++  \
  --workers 32 \
  --batch-size 256  \
  --project marinedebrisdetector \
  --run-name unet++1 \
  --seed 1
```

### Test Model

```
python test.py \
  --data-path /data/marinedebris \
  --ckpt-folder /data/marinedebris/results/
```