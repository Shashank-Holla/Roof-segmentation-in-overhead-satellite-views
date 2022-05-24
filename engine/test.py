import torch
from utils.image_utilities import image_save, denormalize
from utils.metrics import dice_coefficient_metric, rmse_metric
import torch.optim as optim

images_mean = (0.485, 0.456, 0.406)
images_std = (0.229, 0.224, 0.225)

def test(model, device, test_loader, criterion_BCE, criterion_DiceLoss, epoch, metrics_history, kbar=None):
    model.eval()

    # collect stats - epoch_loss(cross entropy loss + dice loss), Dice Coefficient, Root Mean Square error 
    epoch_loss = 0
    dice_coeff = 0
    rms_error = 0

    with torch.no_grad():        
        for batch_id, batch in enumerate(test_loader):
            data = batch['sat_img'].to(device)
            mask_target = batch['gt_img'].to(device)

            mask_pred = model(data)
            segmentation_loss = criterion_BCE(mask_pred, mask_target) + criterion_DiceLoss(mask_pred, mask_target)

            # Metrics calculation
            epoch_loss += segmentation_loss.item()
            dice_coeff += dice_coefficient_metric(mask_pred, mask_target).item()
            rms_error += rmse_metric(mask_pred, mask_target).item()
    
   
   # Average out metrics.   
    epoch_loss /= len(test_loader)
    dice_coeff /= len(test_loader)
    rms_error /= len(test_loader)
    metrics_history["epoch_loss"].append(epoch_loss)
    metrics_history["dice_coeff"].append(dice_coeff)
    metrics_history["rms_error"].append(rms_error)

    if kbar is not None:
        kbar.add(1, values=[("test_loss", epoch_loss), ("test_dice_coeff", dice_coeff), ("test_rmse", rms_error)]) 
          
    print(f"Test set: Epoch Loss: {round(epoch_loss, 4)}\tDice Coeff: {round(dice_coeff, 4)}\tRMS Error: {round(rms_error, 4)}")

        # Save image
    sat_images = denormalize(data[:16].detach().cpu(), images_mean, images_std)
    image_save(sat_images, "test_sat_image"+str(epoch)+"_")
    image_save(mask_target.detach().cpu(), "test_gt"+str(epoch)+"_")
    image_save(mask_pred.detach().cpu(), "test_pred"+str(epoch)+"_")

    return metrics_history