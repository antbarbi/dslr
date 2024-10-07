import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from logreg_model import LogRegModel
import json
import joblib


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=str,
        help="filename of the test dataset"
    )
    parser.add_argument(
        "weights",
        type=str,
        help="filename of the weights file"
    )
    return parser.parse_args()


def main():
    args = parse()
    data = pd.read_csv(args.dataset)
    data.drop([
                "Index", 
                "First Name", 
                "Last Name", 
                "Birthday", 
                "Best Hand",
                "Astronomy",
                "Flying",
                "Charms",
            ],
            axis=1,
            inplace=True)

    imputer = SimpleImputer(strategy='mean')

    X = data.drop("Hogwarts House", axis=1)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X))

    with open(args.weights, "r") as f:
        weights = json.load(f)

    classes = list(weights.keys())
    classifiers = []

    for cls in classes:
        model = LogRegModel(X.shape[1])
        model.set_weights(weights[cls])
        classifiers.append(model)
    
    def one_vs_all(X):
        # List to store scores from each classifier
        scores = np.zeros((X.shape[0], len(classes)))
        
        # For each classifier (one for each class)
        for idx, model in enumerate(classifiers):
            scores[:, idx] = model.predict_proba(X)[:, 1]
            
        # Choose the class with the highest score
        return np.argmax(scores, axis=1)
    
    # Make predictions on the dataset_test
    y_pred = one_vs_all(X)
    
    # Load the label encoder
    le = joblib.load("label_encoder.pkl")
    
    # Convert numeric predictions back to class labels
    y_pred_labels = le.inverse_transform(y_pred)

    predictions = pd.DataFrame({
        "Index": range(len(y_pred_labels)),
        "Hogwarts House": y_pred_labels
    })
    predictions.to_csv("houses.csv", index=False)
    print("Predictions saved to houses.csv")


if __name__ == "__main__":
    main()