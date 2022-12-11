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

we use weights and biases for logging. Make sure, you have an account and can login per console
```bash
pip install wandb
wandb login
```

### Model Training

specify `--download` to download the data to the specified `--data-path`.

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
  --ckpt-folder checkpoints/unet++1
```
