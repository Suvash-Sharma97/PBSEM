import os
import glob
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import opensmile

# ---------- PATHS ----------

RAVDESS_ROOT = "AudioFiles"
OUTPUT_CSV = "ravdess_prosody_full.csv"

# ---------- RAVDESS METADATA PARSING ----------

# RAVDESS filename pattern: MM-VC-EE-II-SS-RR-AA.wav
# 03-01-06-01-02-01-12.wav etc. [web:33][web:41]
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def parse_ravdess_filename(fname):
    base = os.path.splitext(fname)[0]
    parts = base.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename format: {fname}")

    modality = parts[0]       # 01 / 02 / 03...
    vocal_channel = parts[1]  # 01 = speech, 02 = song
    emotion_id = parts[2]     # 01..08
    intensity_id = parts[3]   # 01 = normal, 02 = strong
    statement_id = parts[4]   # 01, 02
    repetition_id = parts[5]  # 01, 02
    actor_id = parts[6]       # 01..24

    emotion_label = EMOTION_MAP.get(emotion_id, "unknown")

    #derive gender from actor ID (odd=male, even=female)
    actor_num = int(actor_id)
    speaker_gender = "M" if actor_num % 2 == 1 else "F"

    return {
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion_id": emotion_id,
        "emotion_label": emotion_label,
        "intensity_id": intensity_id,
        "statement_id": statement_id,
        "repetition_id": repetition_id,
        "actor_id": actor_id,
        "speaker_gender": speaker_gender,
    }

# ---------- PARSELMOUTH (PRAAT) PROSODIC FEATURES ----------

def extract_parselmouth_features(wav_path):
    sound = parselmouth.Sound(wav_path)

    # Pitch
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    pitch_values = pitch.selected_array['frequency']
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

    # Use 5‑argument jitter, 7‑argument shimmer as in current Praat docs. [web:52][web:53][web:54]
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


# ---------- openSMILE EMOBASE FEATURES ----------

# emobase is a standard prosodic feature set for emotion recognition. [web:14][web:20]
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_opensmile_features(wav_path):
    df = smile.process_file(wav_path)
    feat_dict = df.iloc[0].to_dict()
    # Prefix all columns to avoid collisions
    return {f"os_{k}": float(v) for k, v in feat_dict.items()}

# ---------- COLLECT FILES & BUILD DATASET ----------

def collect_ravdess_files(root):
    pattern = os.path.join(root, "Actor_*", "*.wav")
    return glob.glob(pattern)

def build_ravdess_dataset():
    wav_files = collect_ravdess_files(RAVDESS_ROOT)
    if not wav_files:
        print(f"No wav files found in {RAVDESS_ROOT}")
        return

    rows = []

    for idx, wav_path in enumerate(sorted(wav_files)):
        fname = os.path.basename(wav_path)
        print(f"[{idx+1}/{len(wav_files)}] {fname}")

        try:
            meta = parse_ravdess_filename(fname)
        except Exception as e:
            print(f"  [WARN] Skipping {fname}: {e}")
            continue

        # only speech and not songs? keep VC=01.
        if meta["vocal_channel"] != "01":
            continue

        base_info = {
            "file_name": fname,
            "emotion_id": meta["emotion_id"],
            "emotion_label": meta["emotion_label"],
            "actor_id": meta["actor_id"],
            "intensity_id": meta["intensity_id"],
            "statement_id": meta["statement_id"],
            "repetition_id": meta["repetition_id"],
            # Optional: keep gender just as metadata, not as focus
            "speaker_gender": meta["speaker_gender"],
        }

        try:
            praat_feats = extract_parselmouth_features(wav_path)
        except Exception as e:
            print(f"  [ERROR] Parselmouth failed on {fname}: {e}")
            continue

        try:
            os_feats = extract_opensmile_features(wav_path)
        except Exception as e:
            print(f"  [ERROR] openSMILE failed on {fname}: {e}")
            continue

        full_feats = {**base_info, **praat_feats, **os_feats}
        rows.append(full_feats)

    if not rows:
        print("No rows extracted.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved full prosody dataset to {OUTPUT_CSV}")
    print(f"Shape: {df.shape}")  # expect ~1440 rows x (metadata + ~1000+ features)

if __name__ == "__main__":
    build_ravdess_dataset()
