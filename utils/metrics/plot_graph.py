import matplotlib.pyplot as plt
import os
import time

def loss_dice_RMS_metrics_graph(results_dir, train_metrics_dict, test_metrics_dict):
    '''
    Chart Loss, Dice Coefficient, Root Mean Square error for segmentation images for train as well as test.
    Input
    train_metrics_dict - Dictionary with train stats.
    test_metrics_dict - Dictionary with test stats.
    '''
    train_loss = train_metrics_dict["epoch_loss"]
    train_dice = train_metrics_dict["dice_coeff"]
    train_RMS = train_metrics_dict["rms_error"]


    test_loss = test_metrics_dict["epoch_loss"]
    test_dice = test_metrics_dict["dice_coeff"]
    test_RMS = test_metrics_dict["rms_error"]    

  
    fig, axs = plt.subplots(2,2,figsize=(12,12))

    axs[0][0].set_title("Train/Test Loss")
    axs[0][0].plot(train_loss, label = "train loss")
    axs[0][0].plot(test_loss, label = "test loss")
    axs[0][0].set_xlabel("Epoch")
    axs[0][0].set_ylabel("Loss")
    axs[0][0].legend(loc="best")

    axs[0][1].set_title("Train/Test Dice Coefficient")
    axs[0][1].plot(train_dice, label = "train dice coefficient")
    axs[0][1].plot(test_dice, label = "test dice coefficient")
    axs[0][1].set_xlabel("Epoch")
    axs[0][1].set_ylabel("Dice Coefficient")
    axs[0][1].legend(loc="best")

    axs[1][1].set_title("Train/Test Root Mean Square Error (RMSE)")
    axs[1][1].plot(train_RMS, label = "train RMSE")
    axs[1][1].plot(test_RMS, label = "train RMSE")
    axs[1][1].set_xlabel("Epoch")
    axs[1][1].set_ylabel("Root Mean Square Error")
    axs[1][1].legend(loc="best")

    filename = (f'LossDiceRMSEcurve_{time.strftime("%Y-%m-%d")}.jpg')
    filename = os.path.join(results_dir, filename)
    plt.savefig(filename)