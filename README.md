# Roof Segmentation in overhead Satellite views

In this paper, we propose a fully convolutional encoder-decoder architecture to solve image segmentation of planar rooftops. The designed architecture utilizes skip connections in the encoder and decoder blocks which assists in vanishing gradient problem when the network gets deeper.

### Dataset

Download the INRIA dataset from the following site - [INRIA Dataset](https://project.inria.fr/aerialimagelabeling/download/)

Here, the train set is split into train/test sets in ratio 80:20. Please remove existing test folder and execute the following-

```
python3 prepare_data.py './data'
```

### Model architecture

![](/result_files/model.png)

### Install the dependencies

```
pip install -r requirements.txt
```

### Training the model

The following parameters can be passed during model training -

a. model - Name of the model to train. Model available for training presently is- SatUMaskNet.

b. lr - Learning rate

c. epochs - Number of epochs to train the model

d. batch_size - Number of images in a batch to train

e. patch_size - Size of the image patch on which the model is to be trained.

Command to start model training and inferencing -

```
python3 main.py --model SatUMaskNet --lr 0.01 --epochs 24 --batch_size 4 --patch_size 256
```

### Results

![](/result_files/segmentation.png)
