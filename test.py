import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
file = "dataset_train.csv"
data = pd.read_csv(file)

# Prepare the data
# Assuming the target column is named 'target' and the rest are features
X = data.drop(columns=['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Hogwarts House'])
y = data['Hogwarts House']
X = X.fillna(X.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LogisticRegression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Optionally, print the predictions
print("Predictions:", y_pred)
