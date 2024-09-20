import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from logreg_model import LogRegModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving the label encoder


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=str,
        help="filename to get describe for"
    )
    return parser.parse_args()


def main():
    data = pd.read_csv(parse().dataset)
    
    # Preprocess data
    data.drop([
                "Index", 
                "First Name", 
                "Last Name", 
                "Birthday", 
                "Best Hand",
                "Charms",
                "Ancient Runes",
                "Defense Against the Dark Arts",
                "Astronomy",
                "Flying",
            ],
            axis=1,
            inplace=True)

    imputer = SimpleImputer(strategy='mean')

    X = data.drop("Hogwarts House", axis=1)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y = data["Hogwarts House"]

    le = LabelEncoder()
    y = le.fit_transform(data["Hogwarts House"])

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model

    classifiers = []
    classes = np.unique(y)
    weights = {}

    for cls in classes:
        binary_y_train = (y_train == cls).astype(int)

        model = LogRegModel(X_train.shape[1])
        model.fit(X_train, binary_y_train, 0.5, 200)
        classifiers.append(model)

        weights[str(cls)] = model.get_weights()  # Convert keys to strings
    
    with open("weights.json", "w") as f:
        json.dump(weights, f)

    joblib.dump(le, "label_encoder.pkl")
    
    print("Weights and label encoder saved")
    
    # def predict(X):
    #     # List to store scores from each classifier
    #     scores = np.zeros((X.shape[0], len(classes)))
        
    #     # For each classifier (one for each class)
    #     for idx, model in enumerate(classifiers):
    #         # Get the probabilities for the positive class (class == idx)
    #         scores[:, idx] = model.predict_proba(X)[:, 1]  # assuming your LogRegModel has predict_proba
            
    #     # Choose the class with the highest score
    #     return np.argmax(scores, axis=1)
    
    # # Make predictions on the test set
    # y_pred = predict(X_test)
    
    # # Calculate accuracy
    # accuracy = np.mean(y_pred == y_test)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()