import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load preprocessed features and labels
X = np.load("X_features_scaled.npy")  #input features
y = np.load("y_labels.npy")  # integer state_id: 0/1/2

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#a simple but strong baseline for SER: Random Forest
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_val)
print("Validation report:")
print(classification_report(y_val, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred))

#save model for later use in Flask
joblib.dump(clf, "speech_state_model.pkl")
print("Saved model to speech_state_model.pkl")
