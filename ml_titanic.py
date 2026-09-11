import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    log_loss
)

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load Titanic Dataset
# --------------------------------------------------

df = pd.read_csv("data/train.csv")


# --------------------------------------------------
# 2. Select Features and Target
# --------------------------------------------------

features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

X = df[features].copy()
y = df["Survived"]


# --------------------------------------------------
# 3. Handle Missing Values
# --------------------------------------------------

X["Age"] = X["Age"].fillna(X["Age"].median())


# --------------------------------------------------
# 4. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 5. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 6. Logistic Regression
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]


print("\n===== Logistic Regression =====")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
print("Log Loss :", log_loss(y_test, y_prob))

# --------------------------------------------------
# ROC Curve
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob):.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()


plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 7. Logistic Regression Coefficients
# --------------------------------------------------

print("\nCoefficients:")

for feature, coefficient in zip(features, model.coef_[0]):
    print(feature, ":", coefficient)

# --------------------------------------------------
# 8. L1 Regularization
# --------------------------------------------------

l1_model = LogisticRegression(
    solver="liblinear",
    C=1.0,
    l1_ratio=1.0,
    max_iter=1000
)

l1_model.fit(X_train_scaled, y_train)

l1_pred = l1_model.predict(X_test_scaled)

print("\n===== L1 Regularization =====")

print("Accuracy:", accuracy_score(y_test, l1_pred))

print("\nL1 Coefficients:")

for feature, coefficient in zip(features, l1_model.coef_[0]):
    print(feature, ":", coefficient)

# --------------------------------------------------
# 9. L2 Regularization
# --------------------------------------------------

l2_model = LogisticRegression(
    C=1.0,
    max_iter=1000
)

l2_model.fit(X_train_scaled, y_train)

l2_pred = l2_model.predict(X_test_scaled)

print("\n===== L2 Regularization =====")

print("Accuracy:", accuracy_score(y_test, l2_pred))

print("\nL2 Coefficients:")

for feature, coefficient in zip(features, l2_model.coef_[0]):
    print(feature, ":", coefficient)


# --------------------------------------------------
# 10. Elastic Net
# --------------------------------------------------

elastic_model = LogisticRegression(
    solver="saga",
    l1_ratio=0.5,
    C=1.0,
    max_iter=5000
)

elastic_model.fit(X_train_scaled, y_train)

elastic_pred = elastic_model.predict(X_test_scaled)

print("\n===== Elastic Net =====")

print("Accuracy:", accuracy_score(y_test, elastic_pred))

print("\nElastic Net Coefficients:")

for feature, coefficient in zip(features, elastic_model.coef_[0]):
    print(feature, ":", coefficient)


# --------------------------------------------------
# 11. Pipeline + Cross Validation
# --------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\n===== Pipeline Cross Validation =====")

print("CV Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


# --------------------------------------------------
# 12. GridSearchCV - Hyperparameter Tuning
# --------------------------------------------------

param_grid = {
    "logistic__C": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print("\n===== GridSearchCV =====")

print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)


# --------------------------------------------------
# 13. Best Model Evaluation
# --------------------------------------------------

best_model = grid_search.best_estimator_

y_pred_grid = best_model.predict(X_test)

print("\n===== Best Balanced Model =====")

print("Test Accuracy:", accuracy_score(y_test, y_pred_grid))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_grid))

print("\nPrecision:", precision_score(y_test, y_pred_grid))
print("Recall   :", recall_score(y_test, y_pred_grid))
print("F1 Score :", f1_score(y_test, y_pred_grid))