
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def get_classifier(name="knn"):
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=3)
    elif name == "svm":
        return SVC(kernel='rbf')
    else:
        raise ValueError("Unsupported classifier")
