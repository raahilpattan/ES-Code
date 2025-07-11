from sklearn.linear_model import LassoCV
import numpy as np

def lasso_bic_anomaly(error_tensor):
    """
    detects anomalies in the error tensor using lasso regression with cross-validation.
    for each frame, fits a lasso model to identify which features contribute most to the reconstruction error.
    returns an array of anomaly scores with the same shape as error_tensor.

    parameters:
        error_tensor (np.ndarray): 2d array (frames x features) of reconstruction errors

    returns:
        np.ndarray: anomaly scores, same shape as error_tensor
    """
    n_frames, n_features = error_tensor.shape
    anomalies = np.zeros_like(error_tensor)
    for i in range(n_frames):
        X = np.diag(np.abs(error_tensor[i]))
        y = error_tensor[i]
        if np.count_nonzero(y) > 1:
            model = LassoCV(cv=5).fit(X, y)
            anomalies[i] = model.predict(X)
    return anomalies

# alpha is the smoothing factor, typically between 0 and 1
def ewma_monitoring(errors, alpha=0.3):
    """
    computes the exponentially weighted moving average (EWMA) of the error sequence.
    this is used to monitor the process and detect shifts or drifts in error over time.

    parameters:
        errors (np.ndarray): 1d array of reconstruction errors per frame
        alpha (float): smoothing factor between 0 and 1 (default 0.3)

    returns:
        np.ndarray: ewma statistics, same length as errors
    """
    ewma = np.zeros_like(errors)
    ewma[0] = errors[0]
    for t in range(1, len(errors)):
        ewma[t] = alpha * errors[t] + (1 - alpha) * ewma[t-1]
    return ewma