
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(X, y, classifier):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)

    acc = accuracy_score(y_test, preds)

    return {
        "accuracy": acc,
        "report": classification_report(y_test, preds)
    }
