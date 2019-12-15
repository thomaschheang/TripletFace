Triplet loss for facial recognition.

# Triplet Face

The repository contains code for the application of triplet loss training to the
task of facial recognition. This code has been produced for a lecture and is not
going to be maintained in any sort.

# Python Notebook

```bash
!git clone https://github.com/thomaschheang/TripletFace.git

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import csv

plt.ion()  

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
    
!pip3 install triplettorch
from triplettorch import HardNegativeTripletMiner
from triplettorch import AllTripletMiner

from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/My Drive/DatasetIA.zip"

%cd ../content/TripletFace/
!ls

!python -m tripletface.train -s ../dataset/ -m -e 5 -b 64 -i 240

from tripletface.core.model import Encoder

model = Encoder(64)
weights = torch.load( "/content/TripletFace/model/model.pt" )['model']
model.load_state_dict( weights )
jit_model = torch.jit.trace(model,torch.rand(3, 3, 5, 8)) 
torch.jit.save( jit_model, "/content/drive/My Drive/IA/jit_model.pt" )

!cd /content/drive/My\ Drive/ && git init IA/

!cd /content/drive/My\ Drive/IA/ && git add jit_model.pt

u = 'thomaschheang'
p =  'password'
!cd /content/drive/My\ Drive/IA/ && git remote add origin https://{u}:{p}@github.com/{u}/TripletFace.git

!cd /content/drive/My\ Drive/IA/ && git pull origin master
```

![TSNE_Latent](TSNE_Latent.png)

## Architecture

The proposed architecture is pretty simple and does not implement state of the
art performances. The chosen architecture is a fine tuning example of the
resnet18 CNN model. The model includes the freezed CNN part of resnet, and its
FC part has been replaced to be trained to output latent variables for the
facial image input.

The dataset needs to be formatted in the following form:
```
dataset/
| test/
| | 0/
| | | 00563.png
| | | 01567.png
| | | ...
| | 1/
| | | 00011.png
| | | 00153.png
| | | ...
| | ...
| train/
| | 0/
| | | 00001.png
| | | 00002.png
| | | ...
| | 1/
| | | 00001.png
| | | 00002.png
| | | ...
| | ...
| labels.csv        # id;label
```

## Install

Install all dependencies ( pip command may need sudo ):
```bash
cd TripletFace/
pip3 install -r requirements.txt
```

## Usage

For training:
```bash
usage: train.py [-h] -s DATASET_PATH -m MODEL_PATH [-i INPUT_SIZE]
                [-z LATENT_SIZE] [-b BATCH_SIZE] [-e EPOCHS]
                [-l LEARNING_RATE] [-w N_WORKERS] [-r N_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  -s DATASET_PATH, --dataset_path DATASET_PATH
  -m MODEL_PATH, --model_path MODEL_PATH
  -i INPUT_SIZE, --input_size INPUT_SIZE
  -z LATENT_SIZE, --latent_size LATENT_SIZE
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -e EPOCHS, --epochs EPOCHS
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
  -w N_WORKERS, --n_workers N_WORKERS
  -r N_SAMPLES, --n_samples N_SAMPLES
```

## References

* Resnet Paper: [Arxiv](https://arxiv.org/pdf/1512.03385.pdf)
* Triplet Loss Paper: [Arxiv](https://arxiv.org/pdf/1503.03832.pdf)
* TripletTorch Helper Module: [Github](https://github.com/TowardHumanizedInteraction/TripletTorch)

## Todo ( For the students )

**Deadline Decembre 13th 2019 at 12pm**

The students are asked to complete the following tasks:
* Fork the Project
* Improve the model by playing with Hyperparameters and by changing the Architecture ( may not use resnet )
* JIT compile the model ( see [Documentation](https://pytorch.org/docs/stable/jit.html#torch.jit.trace) )
* Add script to generate Centroids and Thesholds using few face images from one person
* Generate those for each of the student included in the dataset
* Add inference script in order to use the final model
* Change README.md in order to include the student choices explained and a table containing the Centroids and Thesholds for each student of the dataset with a vizualisation ( See the one above )
* Send the github link by mail
