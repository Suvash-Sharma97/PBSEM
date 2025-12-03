import os
import tempfile

from flask import Flask, render_template, request
import numpy as np
import joblib
import parselmouth
from parselmouth.praat import call
import opensmile

# ---------------- CONFIG ----------------
ALLOWED_EXTENSIONS = {".wav", ".WAV"}

app = Flask(__name__)

# LOAD PREPROCESSORS AND MODEL CREATED DURING TRAINING 
pca_bundle = joblib.load("pca_os_features.pkl")
os_scaler = pca_bundle["os_scaler"]
pca = pca_bundle["pca"]
PRAAT_FEATURES = pca_bundle["praat_features"]
OS_FEATURE_NAMES = pca_bundle["os_feature_names"]   # WITHOUT 'os_' prefix
STATE_TO_ID = pca_bundle["state_to_id"]
ID_TO_STATE = {v: k for k, v in STATE_TO_ID.items()}

full_scaler = joblib.load("scaler.pkl")
model = joblib.load("speech_state_model_tuned.pkl")

# training means for each openSMILE feature (before scaling)
os_feature_means = os_scaler.mean_

# openSMILE config IDENTICAL to training
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals,
)

#FEATURE EXTRACTION: EXACT COPY OF TRAINING CODE

def extract_parselmouth_features(wav_path):
    sound = parselmouth.Sound(wav_path)

    # Pitch
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]

    if len(pitch_values) > 0:
        f0_mean = np.mean(pitch_values)
        f0_std  = np.std(pitch_values)
        f0_min  = np.min(pitch_values)
        f0_max  = np.max(pitch_values)
    else:
        f0_mean = f0_std = f0_min = f0_max = 0.0

    # Intensity
    intensity = sound.to_intensity(time_step=0.01)
    intensity_values = intensity.values[0]
    intensity_mean = float(np.mean(intensity_values))
    intensity_std  = float(np.std(intensity_values))

    # Point process
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    # Jitter / shimmer (same arguments as dataset script)
    jitter_local = call(
        point_process,
        "Get jitter (local)",
        0, 0, 0.0001, 0.02, 1.3
    )

    shimmer_local = call(
        [sound, point_process],
        "Get shimmer (local)",
        0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    # HNR
    harmonicity = sound.to_harmonicity(time_step=0.01, minimum_pitch=75)
    hnr_values = harmonicity.values[0]
    hnr_mean = float(np.mean(hnr_values))
    hnr_std  = float(np.std(hnr_values))

    return {
        "f0_mean": float(f0_mean),
        "f0_std":  float(f0_std),
        "f0_min":  float(f0_min),
        "f0_max":  float(f0_max),
        "intensity_mean": intensity_mean,
        "intensity_std":  intensity_std,
        "jitter_local":   float(jitter_local),
        "shimmer_local":  float(shimmer_local),
        "hnr_mean":       hnr_mean,
        "hnr_std":        hnr_std,
    }

def extract_opensmile_features_raw(wav_path):
    """
    Returns dict of raw openSMILE emobase features (no 'os_' prefix),
    exactly like in the training script BEFORE adding 'os_'.
    """
    df = smile.process_file(wav_path)
    return df.iloc[0].to_dict()

def build_feature_vector(wav_path):
    #1. PRAAT BLOCK (same keys as training)
    praat_feats = extract_parselmouth_features(wav_path)
    praat_vec = np.array(
        [praat_feats[name] for name in PRAAT_FEATURES],
        dtype=np.float32
    ).reshape(1, -1)

    #2. openSMILE BLOCK: SAME FEATURESR AS TRAINING
    os_dict = extract_opensmile_features_raw(wav_path)

    os_vals = []
    for idx, name in enumerate(OS_FEATURE_NAMES):
        if name in os_dict:
            os_vals.append(float(os_dict[name]))
        else:
            # Fill with training mean if missing
            os_vals.append(float(os_feature_means[idx]))
    os_raw = np.array(os_vals, dtype=np.float32).reshape(1, -1)

    # 3. APPLY SCALER + PCA
    os_scaled = os_scaler.transform(os_raw)
    os_pca = pca.transform(os_scaled)

    #4. CONCATENATE AND APPLY FINAL SCALER
    X = np.hstack([praat_vec, os_pca])
    X_scaled = full_scaler.transform(X)
    return X_scaled

def predict_state(wav_path):
    X = build_feature_vector(wav_path)
    pred_id = int(model.predict(X)[0])
    state = ID_TO_STATE.get(pred_id, "unknown")
    proba = model.predict_proba(X)[0]
    return state, pred_id, proba

#FLASK ROUTES

def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        if not allowed_file(file.filename):
            return render_template("index.html", error="Only .wav files are supported")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            state, state_id, proba = predict_state(tmp_path)
        except Exception as e:
            os.remove(tmp_path)
            return render_template("index.html", error=f"Processing error: {e}")

        os.remove(tmp_path)

        probs = {ID_TO_STATE[i]: float(p) for i, p in enumerate(proba)}

        return render_template("index.html", prediction=state, probs=probs)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
