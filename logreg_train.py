import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from logreg_model import LogRegModel
import joblib  # For saving the label encoder


def parse() -> argparse.Namespace:
    """parse arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=str,
        help="filename to get describe for"
    )
    parser.add_argument(
        "-c",
        action="store_true",
        help="cost history"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-sgd",
        action="store_true",
        help="use stochastic gradient descent"
    )
    group.add_argument(
        "-mbgd",
        nargs="?",
        const=32,
        type=int,
        choices=range(2, 1281),
        metavar="[2-1280]",
        help="use mini-batch gradient descent with batch size between 2 and 1280"
    )

    return parser.parse_args()


def plot_loss(classes, cost_histories, le):
    """plot the loss curve"""

    plt.figure(figsize=(10, 6))
    for cls in classes:
        house_name = le.inverse_transform([cls])[0]
        plt.plot(cost_histories[str(cls)], label=f'{house_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Cost History for Each House')
    plt.legend()
    plt.show()


def main():
    """training phase of data from dataset_train.csv"""

    args = parse()
    data = pd.read_csv(args.dataset)
    
    # Preprocess data
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
    y = data["Hogwarts House"]

    le = LabelEncoder()
    y = le.fit_transform(data["Hogwarts House"])

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    classifiers = []
    classes = np.unique(y)
    weights = {}
    cost_histories = {}

    for cls in classes:
        binary_y_train = (y_train == cls).astype(int)

        model = LogRegModel(X_train.shape[1])
        if args.sgd:
            _, cost_history = model.fit(X_train, binary_y_train, 0.01, 250, "sgd")
        elif args.mbgd:
            _, cost_history = model.fit(X_train, binary_y_train, 0.2, 1000, "mbgd", batch_size=args.mbgd)
        else:
            _, cost_history = model.fit(X_train, binary_y_train, 0.5, 1000, "gd")
        classifiers.append(model)
        weights[str(cls)] = model.get_weights()  # Convert keys to strings
        cost_histories[str(cls)] = cost_history
    
    with open("weights.json", "w") as f:
        json.dump(weights, f)

    joblib.dump(le, "label_encoder.pkl")

    if args.c:
        plot_loss(classes, cost_histories, le)
    
    print("Weights and label encoder saved")
    
    def one_vs_all(X):
        # List to store scores from each classifier
        scores = np.zeros((X.shape[0], len(classes)))
        
        # For each classifier (one for each class)
        for idx, model in enumerate(classifiers):
            # Get the probabilities for the positive class (class == idx)
            scores[:, idx] = model.predict_proba(X)[:, 1]  # assuming your LogRegModel has predict_proba
            
        # Choose the class with the highest score
        return np.argmax(scores, axis=1)
    
    # Make predictions on the test set
    y_pred = one_vs_all(X_test)
    
    # Calculate accuracy on the test set (not the truth!)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
