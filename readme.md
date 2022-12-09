# Marine Debris Detector

<img src="doc/marinedebrisdetector.png" width=800px>

## Getting Started

package installation
```
pip install git+https://github.com/marccoru/marinedebrisdetector
```

model prediction of qualitative test images.
```
marinedebrisdetector --plot-qualitative
```

prediction of a Sentinel-2 scene
```
# download example scene
marinedebrisdetector --download-durban

# predict the durban example scene
marinedebrisdetector durban_20190424.tif
```

for more options call `marindebrisdetector --help`

## Detailed Setup 

### Python Environment
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Datasets

more details in a dedicated [data page](doc/data.md)

### Pre-trained Models

more details in a dedicated [models page](doc/models.md)

### Train Model

see full training scripts in [training scripts](training_scripts) folder.
We use weights and biases for logging. Check the previous training runs [here](https://wandb.ai/marccoru/marinedebrisdetector)

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

#### our Marine Debris Detector
```
python test.py --ckpt-folder /data/marinedebris/results/ours/unet++_2022-10-21-1e6
```

#### Mifdal et al., 2021
```
python test.py --comparison mifdal --ckpt-folder /data/marinedebris/results/mifdal/unet-posweight1-lr001-bs160-ep50-aug1-seed0
```

#### Kikaki et al., 2022
```
python test_kikaki.py --ckpt-folder /data/marinedebris/results/kikaki/randomforest
```

