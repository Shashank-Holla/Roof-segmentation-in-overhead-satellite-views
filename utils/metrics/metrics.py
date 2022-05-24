import torch
import torch.nn as nn

# dice coefficient
def dice_coefficient_metric(yhat, y):
    '''
    Measure to check the similarity between the two samples.
    dice_coefficient = 2 * area of overlap / (total number of pixels in both images.)
    Input-
    yhat - Tensor (N x C x H x W) - Model's prediction
    y - Tensor (N x C x H x W) - Ground truth to be compared with.
    Output-
    dice_coefficient - score of the similarity between the samples. range: [0,1]
    '''
    # apply sigmoid to bring it to [0,1] range.
    yhat = torch.sigmoid(yhat)
    intersection = torch.sum(yhat * y, (1,2,3))
    sum_cardinalities = torch.sum(yhat + y, (1,2,3))
    dice_coefficient = (2.*intersection)/ sum_cardinalities

    return torch.mean(dice_coefficient)


def rmse_metric(yhat, y):
    '''
    Root mean square error of the two samples. Higher the metric, higher the error.
    Input-
    yhat - Tensor (N x C x H x W) - Model's prediction
    y - Tensor (N x C x H x W) - Ground truth to be compared with.
    Output-
    rms_error - Root mean square error. 
    '''
    yhat = torch.sigmoid(yhat)
    mse_loss = nn.MSELoss()
    rms_error = torch.sqrt(mse_loss(yhat, y))
    return rms_error