import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

import glob
import os
import cv2
import time
import numpy as np
import argparse

from data.dataset.satelliteDataset import SatelliteDataset
from data.transform.albumentation_transform import albumentation_transform
from models.SatUMaskNet import SatUMaskNet
from models.loss.dice_loss import DiceLoss
from engine.train import train
from engine.test import test
from utils.metrics.plot_graph import loss_dice_RMS_metrics_graph
from utils.pkbar import Kbar


data_path = './data'
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'test')
if not os.path.exists("results"):
    os.mkdir("results")
results_dir = 'results'

# results path
filename = (f'best_model_checkpoint_{time.strftime("%Y-%m-%d")}.pt')
model_checkpoint = os.path.join(results_dir, filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="SatUMaskNet", choices=['SatUMaskNet'], help='model to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=24, help='number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=4, help='size of each batch to run')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size on which to train model')
    opt = parser.parse_args()

    # enables the inbuilt CUDNN auto-tuner to speed up training process
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model_name = opt.model
    lr = opt.lr
    EPOCHS = opt.epochs
    batch_size = opt.batch_size
    patch_size = opt.patch_size

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory':True}

    print(f"Prepare training with: Epochs: {EPOCHS}, Learning Rate: {lr} Batch size: {batch_size} and Patch size: {patch_size}")

    train_dataset = SatelliteDataset(train_path, patch_size, tile_size=1000, transform = albumentation_transform)
    train_loader = DataLoader(train_dataset, **params)
    test_dataset = SatelliteDataset(test_path, patch_size, tile_size=1000, transform = albumentation_transform)
    test_loader = DataLoader(test_dataset, **params)
    
    print("Train/Test Dataloader is created")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device:", device)

    if model_name == "SatUMaskNet":
        model = SatUMaskNet(in_channels=3, out_channels=1).to(device)
        print("SatUMaskNet model is loaded")

    # Optimization algorithm from torch.optim
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Loss functions
    # Cross entropy loss 
    criterion_BCE = nn.BCEWithLogitsLoss()
    criterion_DiceLoss = DiceLoss()
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True, threshold=0.001, cooldown=0, min_lr=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=0.03, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    best_loss = np.inf

    metrics_train_history = {"epoch_loss": [], "dice_coeff": [], "rms_error": []}
    metrics_test_history = {"epoch_loss": [], "dice_coeff": [], "rms_error": []}

    print(f'Training start time: {time.strftime("%B %d,%Y %H:%M:%S")}')
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch}")

        kbar = Kbar(target=len(train_loader), epoch=epoch, num_epochs=EPOCHS, width=8, always_stateful=False)

        metrics_train_history = train(model, device, train_loader, optimizer, criterion_BCE, criterion_DiceLoss, epoch, metrics_train_history, kbar, scheduler)
        metrics_test_history = test(model, device, test_loader, criterion_BCE, criterion_DiceLoss, epoch, metrics_test_history, kbar)
        # scheduler step for onCycleLR is done for every batch and will not be done here. For other scheduler methods, scheduler step will be done every epoch.
        if not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step(metrics_train_history["epoch_loss"][-1])
        # Saving the model on the first run and when the test loss is the lowest
        if metrics_test_history["epoch_loss"][-1] < best_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': metrics_test_history["epoch_loss"][-1]
            }, model_checkpoint)
            best_loss = metrics_test_history["epoch_loss"][-1]
    
    # Draw plots
    loss_dice_RMS_metrics_graph(results_dir, metrics_train_history, metrics_test_history)
    
    print(f'Training end time: {time.strftime("%B %d,%Y %H:%M:%S")}')
    print(f"time taken in seconds: {round(time.time()-start_time, 4)}")



