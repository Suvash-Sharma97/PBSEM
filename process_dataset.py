import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

INPUT_CSV = "ravdess_prosody_full.csv"

# Outputs for later training / Flask
OUTPUT_FEATURES = "X_features.npy"
OUTPUT_LABELS = "y_labels.npy"
OUTPUT_META = "meta.csv"          # file_name + original labels
OUTPUT_PCA = "pca_os_features.pkl"
OUTPUT_SCALER = "scaler.pkl"

# Map original RAVDESS emotions into 3–4 states
EMOTION_TO_STATE = {
    "neutral":   "normal",
    "calm":      "normal",
    "happy":     "excited",
    "angry":     "excited",
    "surprised": "excited",
    "sad":       "demotivated",
    "fearful":   "demotivated",
    "disgust":   "demotivated",
}

STATE_TO_ID = {
    "normal": 0,
    "excited": 1,
    "demotivated": 2
}

# Core hand-picked prosodic features from Praat
PRAAT_FEATURES = [
    "f0_mean", "f0_std", "f0_min", "f0_max",
    "intensity_mean", "intensity_std",
    "jitter_local", "shimmer_local",
    "hnr_mean", "hnr_std",
]

# Number of PCA components for openSMILE part
N_PCA_COMPONENTS = 64

def main():
    df = pd.read_csv(INPUT_CSV)

    # 1) Build state labels
    df["state_label"] = df["emotion_label"].map(EMOTION_TO_STATE)
    df = df.dropna(subset=["state_label"])  # drop rows without mapping

    df["state_id"] = df["state_label"].map(STATE_TO_ID)

    # 2) Split columns: metadata, Praat, and openSMILE
    meta_cols = [
        "file_name",
        "emotion_id",
        "emotion_label",
        "actor_id",
        "intensity_id",
        "statement_id",
        "repetition_id",
        "speaker_gender",
        "state_label",
        "state_id",
    ]

    # all columns starting with "os_" are openSMILE emobase features
    os_cols = [c for c in df.columns if c.startswith("os_")]

    # ensure Praat features exist
    missing_praat = [c for c in PRAAT_FEATURES if c not in df.columns]
    if missing_praat:
        raise ValueError(f"Missing expected prosodic features: {missing_praat}")

    # ------------------------------------------------------------------
    # 3) Prepare feature matrices
    # ------------------------------------------------------------------
    praat_X = df[PRAAT_FEATURES].values    # small set (10 features)
    os_X = df[os_cols].values             # large set (~988 features)

    # Standardize openSMILE features before PCA
    os_scaler = StandardScaler()
    os_X_scaled = os_scaler.fit_transform(os_X)

    # PCA reduction for openSMILE block
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    os_pca = pca.fit_transform(os_X_scaled)

    # Concatenate: [Praat features | PCA(openSMILE)]
    X = np.hstack([praat_X, os_pca])

    # Target labels
    y = df["state_id"].values

    # 4) Save outputs
    np.save(OUTPUT_FEATURES, X)
    np.save(OUTPUT_LABELS, y)

    # Keep meta (file_name + labels) for analysis
    df_meta = df[["file_name", "emotion_label", "state_label", "state_id"]]
    df_meta.to_csv(OUTPUT_META, index=False)

    # Save PCA and scaler
    joblib.dump(
        {
            "os_scaler": os_scaler,
            "pca": pca,
            "praat_features": PRAAT_FEATURES,
            "os_feature_names": os_cols,
            "state_to_id": STATE_TO_ID,
        },
        OUTPUT_PCA,
    )

    full_scaler = StandardScaler()
    X_scaled = full_scaler.fit_transform(X)
    np.save("X_features_scaled.npy", X_scaled)
    joblib.dump(full_scaler, OUTPUT_SCALER)

    print("X shape (unscaled / scaled):", X.shape, X_scaled.shape)
    print("Label distribution:", {lbl: int((y == i).sum()) for lbl, i in STATE_TO_ID.items()})

if __name__ == "__main__":
    main()
