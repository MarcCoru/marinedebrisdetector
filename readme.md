# Marine Debris Detector

<img src="doc/marinedebrisdetector.png" width=600px>

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

# predict the durban example scene (takes 11 minutes on cpu Macbook Pro)
marinedebrisdetector durban_20190424.tif
```

for more options call `marindebrisdetector --help`

### Datasets

more details in a dedicated [data page](doc/data.md)

### Pre-trained Models

pretrained segmentation models can be loaded via the torch hub
```

```

a detailed list of weights can be found on the [models page](doc/models.md)

### Model Training

more details in a dedicated [model training page](doc/model_training.md)