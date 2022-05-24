import torch
import torch.optim as optim

from utils.image_utilities import image_save, denormalize
from utils.metrics import dice_coefficient_metric, rmse_metric

images_mean = (0.485, 0.456, 0.406)
images_std = (0.229, 0.224, 0.225)


def train(model, device, train_loader, optimizer, criterion_BCE, criterion_DiceLoss, epoch, metrics_history, kbar=None, scheduler=None):
    """
    model
    """
    model.train()

    # collect stats - epoch_loss(cross entropy loss + dice loss), Dice Coefficient, Root Mean Square error
    epoch_loss = 0
    dice_coeff = 0
    rms_error = 0


    for batch_id, batch in enumerate(train_loader):
        data = batch['sat_img'].to(device)
        mask_target = batch['gt_img'].to(device)

        optimizer.zero_grad()

        # Gather prediction and calculate loss + backward pass + optimize weights
        mask_pred = model(data)
        # Loss calculation: Binary Cross Entropy + Dice Loss
        segmentation_loss = criterion_BCE(mask_pred, mask_target) + criterion_DiceLoss(mask_pred, mask_target)
       
        # Calculate gradients
        segmentation_loss.backward()
        # Optimizer
        optimizer.step()

        #Scheduler update. For one cycle policy, scheduler is to be updated after every batch.
        if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # Metrics calculation, detach from graph
        with torch.no_grad():
            dice_coeff_batch = dice_coefficient_metric(mask_pred, mask_target).item()
            rms_error_batch = rmse_metric(mask_pred, mask_target).item()
            dice_coeff += dice_coeff_batch
            rms_error += rms_error_batch

        epoch_loss += segmentation_loss.item()

        if kbar is not None:
            kbar.update(batch_id, values=[("loss", segmentation_loss), ("dice_coeff", dice_coeff_batch), ("rmse", rms_error_batch)])

    # Average out metrics.   
    epoch_loss /= len(train_loader)
    dice_coeff /= len(train_loader)
    rms_error /= len(train_loader)
    metrics_history["epoch_loss"].append(epoch_loss)
    metrics_history["dice_coeff"].append(dice_coeff)
    metrics_history["rms_error"].append(rms_error)

    print(f"Train set: Epoch Loss: {round(epoch_loss, 4)}\tDice Coeff: {round(dice_coeff, 4)}\tRMS Error: {round(rms_error, 4)}")

    # Save image
    sat_images = denormalize(data[:16].detach().cpu(), images_mean, images_std)
    image_save(sat_images, "train_sat_image"+str(epoch)+"_")
    image_save(mask_target.detach().cpu(), "train_gt"+str(epoch)+"_")
    image_save(mask_pred.detach().cpu(), "train_pred"+str(epoch)+"_")

    return metrics_history