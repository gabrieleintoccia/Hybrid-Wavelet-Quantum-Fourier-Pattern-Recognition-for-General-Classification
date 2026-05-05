import numpy as np
from pathlib import Path

from src.data_loader import load_mat_dataset
from src.preprocessing import wavelet_reduce
from src.quantum_features import qft_features
from src.classical_features import fft_features, wavelet_features
from src.classifiers import get_classifier
from src.evaluation import evaluate_model

# ==============================
# CONFIGURAZIONE DIRETTA
# ==============================
DATASET_PATH = Path(
    " "
)

METHOD = "hybrid"      # qft / fft / wavelet / hybrid
CLASSIFIER = "svm"     # knn / svm


# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(X, method):
    features = []

    for idx, sample in enumerate(X):
        try:
            reduced = wavelet_reduce(sample)

            if method == "qft":
                feat = qft_features(reduced)

            elif method == "fft":
                feat = fft_features(reduced)

            elif method == "wavelet":
                feat = wavelet_features(reduced)

            elif method == "hybrid":
                feat = np.concatenate([
                    wavelet_features(reduced),
                    qft_features(reduced)
                ])
            else:
                raise ValueError(f"Unknown method: {method}")

            features.append(feat)

            if idx % 50 == 0:
                print(f"Sample {idx}/{len(X)} processed...")

        except Exception as e:
            print(f"Sample error {idx}: {e}")

    return np.array(features)


# ==============================
# MAIN
# ==============================
def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset non trovato: {DATASET_PATH}")

    print("Caricamento dataset...")
    X, y = load_mat_dataset(DATASET_PATH)

    print(f"Dataset loaded: {X.shape}, Labels: {y.shape}")

    print("features extraction...")
    features = extract_features(X, METHOD)

    print(f"Features shape: {features.shape}")

    clf = get_classifier(CLASSIFIER)

    print("Valutazione modello...")
    results = evaluate_model(features, y[:len(features)], clf)

    print("\n===== RESULTS =====")
    print(f"Method: {METHOD}")
    print(f"Classifier: {CLASSIFIER}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(results["report"])


if __name__ == "__main__":
    main()
