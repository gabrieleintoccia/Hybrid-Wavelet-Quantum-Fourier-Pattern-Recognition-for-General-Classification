import scipy.io as sio
import numpy as np


def load_mat_dataset(path):
    data = sio.loadmat(path)

    possible_feature_keys = [
        'fea', 'X', 'data', 'images', 'dataset'
    ]

    possible_label_keys = [
        'gnd', 'Y', 'labels', 'target', 'label'
    ]

    feature_key = next((k for k in possible_feature_keys if k in data), None)
    label_key = next((k for k in possible_label_keys if k in data), None)

    print("Keys available in the dataset:", list(data.keys()))

    if feature_key is None:
        raise KeyError(
            f"No feature keys found. Available: {list(data.keys())}"
        )

    if label_key is None:
        raise KeyError(
            f"No label keys found. Available: {list(data.keys())}"
        )

    X = np.array(data[feature_key])
    y = np.array(data[label_key]).flatten()

    return X, y
