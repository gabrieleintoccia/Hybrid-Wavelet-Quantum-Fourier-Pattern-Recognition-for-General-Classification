
import numpy as np
import pywt

def fft_features(vec):
    fft_vals = np.fft.fft(vec)
    return np.abs(fft_vals)

def wavelet_features(vec, wavelet='haar', level=3):
    coeffs = pywt.wavedec(vec, wavelet, level=level)
    return np.concatenate(coeffs)

def hybrid_features(vec):
    return np.concatenate([wavelet_features(vec), fft_features(vec)])
