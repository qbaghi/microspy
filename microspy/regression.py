import numpy as np
from scipy import linalg as la


def least_squares(mat, y):

    return la.pinv(mat.conjugate().transpose().dot(mat)).dot(mat.conjugate().transpose().dot(y))


def generalized_least_squares(mat_dft, y_dft, psd):
    mat_w = np.array([mat_dft[:, j] / psd for j in range(mat_dft.shape[1])]).T
    # Inverse normal matrix
    ZI = la.pinv(np.dot(np.transpose(mat_dft).conj(), mat_w))
    return ZI.dot(np.transpose(mat_w).conj().dot(y_dft))


def gsl_covariance(mat_dft, psd):

    mat_dft_normalized = mat_dft / np.sqrt(mat_dft.shape[0])
    mat_w = np.array([mat_dft_normalized[:, j] / psd
                      for j in range(mat_dft_normalized.shape[1])]).T

    return la.pinv(np.dot(np.transpose(mat_dft_normalized).conj(), mat_w))
