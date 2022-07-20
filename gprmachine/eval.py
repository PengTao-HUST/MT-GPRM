import numpy as np


def RMSE(Y_test, pred_Y):
    rmse = np.sqrt(((pred_Y - Y_test) ** 2).mean())
    return round(rmse, 4)


def MAE(Y_test, pred_Y):
    error = pred_Y - Y_test
    abs_error = abs(error)
    mae = sum(abs_error)/len(abs_error)
    return round(mae, 4)


def PCC(Y_test, pred_Y):
    pcc = np.corrcoef(Y_test, pred_Y)
    return round(pcc[0,1], 4)


def MAE_RMSE_PCC(Y_test, pred_Y):
    """
        This function is for evaluating the performance of the method.
    """
    pred_Y = np.asarray(pred_Y)
    mae = MAE(Y_test, pred_Y)
    rmse = RMSE(Y_test, pred_Y)
    pcc = PCC(Y_test, pred_Y)
    return [mae, rmse, pcc]
