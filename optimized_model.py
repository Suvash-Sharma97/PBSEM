import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import joblib

# Load features and labels
X = np.load("X_features_scaled.npy")
y = np.load("y_labels.npy")

print("Data shape:", X.shape, "Labels shape:", y.shape)

# Base model
base_clf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

# Hyperparameter search space (kept small for speed but effective) [web:100][web:111]
param_dist = {
    "n_estimators": randint(200, 600),
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_clf,
    param_distributions=param_dist,
    n_iter=30,                 # increase to 50+ if you want more thorough search
    scoring="f1_macro",
    n_jobs=-1,
    cv=cv,
    verbose=2,
    random_state=42,
)

print("Starting RandomizedSearchCV...")
search.fit(X, y)

print("\nBest params:", search.best_params_)
print("Best CV macro F1:", search.best_score_)

best_clf = search.best_estimator_

# Evaluate best model on a held‑out split (20%) to see realistic performance
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_val)

print("\nValidation report with tuned model:")
print(classification_report(y_val, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred))

# Save tuned model
joblib.dump(best_clf, "speech_state_model_tuned.pkl")
print("\nSaved tuned model to speech_state_model_tuned.pkl")
