from flask import Flask, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from ucimlrepo import fetch_ucirepo
import numpy as np

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
@app.route('/')
def home():
    # Load the dataset and split the data
    spambase = fetch_ucirepo(id=94)
    X = spambase.data.features
    y = spambase.data.targets
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create and train the logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, np.ravel(y_train))  # Ensure y_train is flattened
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    # Prepare the confusion matrix as a dictionary for easy access
    confusion_matrix_dict = {
        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1]
    }
    # Render the result in the HTML template
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion_matrix_dict
    }
    return render_template('result.html', results=results)
if __name__ == '__main__':
    app.run(debug=True)
