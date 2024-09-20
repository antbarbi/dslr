import argparse
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
    model = LogRegModel(X_train.shape[1])
    for lr in np.linspace(0.001, 0.5, 100):  # Explore learning rates from 0.001 to 1
            loss = model.fit(X_train, y_train, lr, 200)
            print(loss)
            accuracy = model.score(X_test, y_test)
            print(f"Accuracy: {accuracy * 100:.2f}%")


    ##### Test with sklearn's LogisticRegression
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(multi_class='multinomial', max_iter=4000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":
    main()
