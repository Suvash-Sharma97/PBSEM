import numpy as np
import joblib

# Load what you already created earlier
X_scaled = np.load("X_features_scaled.npy")
y = np.load("y_labels.npy")
pca_bundle = joblib.load("pca_os_features.pkl")
full_scaler = joblib.load("scaler.pkl")

# Just repack into one object for the Flask app
preprocessor = {
    "os_scaler": pca_bundle["os_scaler"],
    "pca": pca_bundle["pca"],
    "praat_features": pca_bundle["praat_features"],
    "os_feature_names": pca_bundle["os_feature_names"],
    "full_scaler": full_scaler,
    "state_to_id": pca_bundle["state_to_id"],
}

joblib.dump(preprocessor, "preprocessor.pkl")
print("Saved preprocessor.pkl")
