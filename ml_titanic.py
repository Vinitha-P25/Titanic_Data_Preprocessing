import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# 1. Load the Titanic dataset
df = pd.read_csv("data/train.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# 2. Select Features
X = df[["Pclass", "Age", "SibSp", "Parch", "Fare"]]


# 3. Select Target
y = df["Survived"]


# 4. Handle missing values
X = X.copy()
X["Age"] = X["Age"].fillna(X["Age"].median())


# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# 6. Create the model
model = LogisticRegression(max_iter=1000)


# 7. Train the model
model.fit(X_train, y_train)

print("Model training completed!")


# 8. Make predictions
y_pred = model.predict(X_test)


# 9. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)


# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 11. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Training predictions
y_train_pred = model.predict(X_train)

# Training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

# Testing accuracy
test_accuracy = accuracy_score(y_test, y_pred)

print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Make predictions
y_pred = model.predict(X_test)

# Training predictions
y_train_pred = model.predict(X_train)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
