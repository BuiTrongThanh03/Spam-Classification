from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ucimlrepo import fetch_ucirepo

def load_data():
    """Fetches and prepares the dataset for training and testing."""
    spambase = fetch_ucirepo(id=94)
    X = spambase.data.features
    y = spambase.data.targets
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate():
    """Trains the model, makes predictions, and evaluates the results."""
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Initialize and train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Package results in a dictionary
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1]
        }
    }
    
    return results
