import torch
from torchvision import utils as torchutils
import os
import time

def image_save(tensor, filename, nrow=4, pad_value=255):
    '''
    tensor - (Denormalized) Tensors of images to be displayed (required format: N x C x H x W)
    filename - Name of the file to be saved as.
    nrow - Number of rows of the montage.
    border_colour - Colour of the border between images. Default=255
    '''
    torchutils.save_image(tensor, fp=os.path.join("results/"+str(filename)+time.strftime("%Y%m%d-%H%M%S")+".jpg"),nrow =nrow, pad_value=pad_value)


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)