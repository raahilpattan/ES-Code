import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker


def cp_decomposition(tensor, rank=3):
    """
    performs CP (CANDECOMP/PARAFAC) decomposition on a 2d tensor (matrix).
    returns the factor matrices for the specified rank.

    parameters:
        tensor (np.ndarray): input data matrix (frames x features)
        rank (int): number of components for decomposition

    returns:
        list: [A, B] factor matrices
    """
    tensor_tl = tl.tensor(tensor)
    cp_tensor = parafac(tensor_tl, rank=rank)
    return cp_tensor.factors  # returns (A, B)


def tucker_decomposition(tensor, ranks):
    """
    performs tucker decomposition on the input tensor.
    returns the core tensor and the factor matrices.

    parameters:
        tensor (np.ndarray): input data matrix
        ranks (tuple): rank for each mode

    returns:
        tuple: (core tensor, list of factor matrices)
    """
    tensor_tl = tl.tensor(tensor)
    core, factors = tucker(tensor_tl, ranks=ranks)
    return core, factors

def reconstruct_cp(factors):
    """
    reconstructs the original matrix from CP factor matrices.

    parameters:
        factors (list): [A, B] factor matrices

    returns:
        np.ndarray: reconstructed matrix (frames x features)
    """
    A, B = factors  # factors[0] is time mode, factors[1] is feature mode
    R = A.shape[1]
    reconstructed = np.zeros((A.shape[0], B.shape[0]))
    for r in range(R):
        outer = np.outer(A[:, r], B[:, r])
        reconstructed += outer
    return reconstructed

def compute_reconstruction_error(original, reconstructed):
    """
    computes the per-frame reconstruction error (L2 norm) between the original and reconstructed matrices.

    parameters:
        original (np.ndarray): original feature matrix
        reconstructed (np.ndarray): reconstructed matrix

    returns:
        np.ndarray: 1d array of errors, one per frame
    """
    return np.linalg.norm(original - reconstructed, axis=1)  # per frame error