import numpy as np
import pywt


def infer_image_shape(length):
    side = int(np.sqrt(length))
    if side * side != length:
        raise ValueError(
            f"Unable to insert square image from length vector {length}"
        )
    return (side, side)


def wavelet_reduce(img_flat, level=2, wavelet='haar'):
    shape = infer_image_shape(len(img_flat))
    img = img_flat.reshape(shape)

    current = img.astype(float)

    for _ in range(level):
        current, _ = pywt.dwt2(current, wavelet)

    return current.flatten()


def normalize_amplitudes(vec):
    vec = np.array(vec, dtype=float)
    norm = np.linalg.norm(vec)

    if norm == 0:
        return vec

    return vec / norm


def pad_to_power_of_two(vec):
    target = 2 ** int(np.ceil(np.log2(len(vec))))
    padded = np.zeros(target)
    padded[:len(vec)] = vec
    return padded
