# Large Scale Detection of Marine Debris with Sentinel-2

> Rußwurm M, Venkatesa J. S., Tuia D. (2023; in preparation) Large-scale Detection of Marine Debris in Coastal Areas with Sentinel-2

<img src="doc/marinedebrisdetector.png" width=600px>


## Getting Started

We provide a python package for easy installation and model predictions

package installation
```
pip install marinedebrisdetector

# alternative directly from GitHub
pip install git+https://github.com/marccoru/marinedebrisdetector
```

model prediction of qualitative test images.
```
marinedebrisdetector --plot-qualitative
```

prediction of a Sentinel-2 scene
```
# download example scene (~300 MB)
marinedebrisdetector --download-durban

# predict the durban example scene
# (15 seconds on GeForce GTX 960/11 minutes on Macbook Pro CPU)
marinedebrisdetector durban_20190424.tif
```

for more options call `marindebrisdetector --help`

### Datasets

We aggregated a combination of existing datasets for training [FloatingObjects Dataset (Mifdal et al., 2020)](https://github.com/ESA-PhiLab/floatingobjects), [Marine Debris Archive (Kikaki et al., 2022)](https://marine-debris.github.io/), [S2Ships (Ciocarlan et al., 2021)](https://github.com/alina2204/contrastive_SSL_ship_detection),
and newly annotated a refinedFloatingObjects dataset and Sentinel-2 images of the [Plastic Litter Projects (Papageorgiou et al., 2022; under review)](https://plp.aegean.gr/)

More details in a dedicated [data page](doc/data.md). 
Executing the training script ([see this "model training" page](doc/model_training.md)) with `--download` will automatically download and uncompress the required datasets (116 GB (uncompressed)).

### Pre-trained Models

We provide pre-trained weights for 12-channel Sentinel-2 imagery.
A detailed list of weights can be found on the [models page](doc/models.md)

pretrained segmentation models can be loaded via the torch hub in python
```python
import torch

torch.hub.load("marccoru/marinedebrisdetector", "unetpp")
torch.hub.load("marccoru/marinedebrisdetector", "unet")

# trained without label refinement (can lead to thinner more fine-grained predictions)
torch.hub.load("marccoru/marinedebrisdetector", "unetpp", label_refinement=False)
```



### Model Training

We provide a [a training script](marinedebrisdetector/train.py) powered by [Pytorch Lightning](https://www.pytorchlightning.ai/) and [Weights and Biases](https://wandb.ai/site) to train new models and reproduce our results.
More details on training commands and ablations in a dedicated page for [model training](doc/model_training.md)
