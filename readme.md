# Prosody Based Student Engagement Monitoring(PBSEM):

Detect the overall state of a student (normal, excited, bored) from a speech clip using prosodic features (pitch, intensity, jitter, shimmer, etc.) and a machine‑learning model.

The project includes a **pretrained model**, so you can run the web app directly. If you want, you can also rebuild the dataset from the RAVDESS and retrain the model yourself.

---

## 1. Overview

This project:

- Lets you upload a `.wav` file through a simple web page.
- Extracts prosodic features using **Praat/Parselmouth** and **openSMILE**.
- Uses a tuned **RandomForest** model to predict:
  - `normal`
  - `excited`
  - `bored`
- Additionally, provides scripts to:
  - Build a feature dataset from RAVDESS.
  - Preprocess features (scaling + PCA).
  - Train and evaluate models.
  - Serve predictions via a Flask web app.

---

## 2. Folder / file structure (key items)

- `frontend.py` – Flask web app entry point (dashboard + prediction).
- `templates/` – HTML templates for the upload and results page.
- `process_dataset.py` – Extracts features from `AudioFiles/Actor_*/*.wav` and builds `ravdess_prosody_full.csv`.
- `export_preprocessor.py` – Exports the preprocessing pipeline (scalers, PCA, feature lists) into `preprocessor.pkl` (and/or `pca_os_features.pkl` and `scaler.pkl`).
- `train_model.py` – Baseline model training script.
- `optimized_model.py` / `train_model_cv.py` – Tuned model training using cross‑validation and hyperparameter search.
- `preprocessor.pkl`, `pca_os_features.pkl`, `scaler.pkl` – Saved preprocessing objects used by training and by the web app.
- `speech_state_model_tuned.pkl` – Pretrained, tuned model used by `frontend.py`.
- `X_features.npy`, `X_features_scaled.npy`, `y_labels.npy` – Preprocessed feature matrices and labels (produced during preprocessing).

> The **RAVDESS audio files are not included**. To rebuild the dataset or retrain the model, you must download RAVDESS separately and place it under `AudioFiles/Actor_*/`.
For that, go to https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio and download the dataset as a zip file. Unzip the folder and name it "AudioFiles".
---

## 3. Quick start – run the web app with the pretrained model

These instructions are for someone who just wants to run the project and see predictions.

### 3.1 Clone the repository

git clone https://github.com/Suvash-Sharma97/PBSEM.git
cd PBSEM

### 3.2 Create and activate a virtual environment

**Windows (PowerShell):**

python -m venv .venv
..venv\Scripts\Activate.ps1

**Linux / macOS (bash/zsh):**

python3 -m venv .venv
source .venv/bin/activate


### 3.3 Install required packages

A `requirements.txt` file is provided. Install everything with:

pip install --upgrade pip
pip install -r requirements.txt

This installs Flask, scikit‑learn, numpy, pandas, parselmouth, opensmile, and other dependencies.

### 3.4 Run the web dashboard

python frontend.py

Then open your browser at:

http://127.0.0.1:5000

Sample Output
![Sample Output](/images/output.jpg)

Steps in the UI:

1. Click **“Choose File”** and select a `.wav` file (mono, 16‑bit PCM recommended).
2. Click **“Predict”**.
3. The page shows:
   - The **predicted state** (normal / excited / demotivated).
   - The **probability** for each state.

At this point you are using the **pretrained model** that ships with the repo (`speech_state_model_tuned.pkl`).

---

## 4. Full pipeline – from raw RAVDESS audio to your own model

If you want to regenerate the dataset and train your own model, follow these steps.

### 4.1 Prepare the RAVDESS audio

1. Download the RAVDESS **speech audio** from kaggle as mentioned before.
2. Create the following structure in the project root:

AudioFiles/
Actor_01/
*.wav
Actor_02/
*.wav
...
Actor_24/
*.wav


Each file name should follow (Usually, it comes by default, so no need to worry):

MM-VC-EE-II-SS-RR-AA.wav

(example: `03-01-03-01-01-01-01.wav`).

### 4.2 Build the prosody feature dataset

From the project root (with the virtual environment activated):

python process_dataset.py


What this does:

- Walks through `AudioFiles/Actor_*/*.wav`.
- Parses metadata from the RAVDESS file names (emotion id, actor id, etc.).
- Extracts:
  - Core prosodic features with **Parselmouth** (F0 stats, intensity stats, jitter, shimmer, HNR).
  - High‑level prosodic features with **openSMILE** (emobase feature set).
- Creates:

ravdess_prosody_full.csv

(one row per audio file, many feature columns).

### 4.3 Preprocess features (scaling, PCA, labels)

Run:

python export_preprocessor.py

This will:

- Load `ravdess_prosody_full.csv`.
- Map detailed RAVDESS emotions to broader states, e.g.:

  - `neutral`, `calm` → **normal**
  - `happy`, `angry`, `surprised` → **excited**
  - `sad`, `fearful`, `disgust` → **demotivated**

- Split metadata from numeric feature columns.
- Standardize features and apply **PCA** to compress the many openSMILE features to a smaller set of components(precisely 988 to 64).
- Save:

  - `X_features.npy`, `X_features_scaled.npy` – feature matrices.
  - `y_labels.npy` – encoded labels for each row.
  - `pca_os_features.pkl` – holds:
    - `os_scaler` (StandardScaler for raw openSMILE features),
    - `pca` (PCA model),
    - `praat_features` (list of Parselmouth feature names),
    - `os_feature_names` (list of raw openSMILE feature names),
    - `state_to_id` (mapping from state label → integer id).
  - `scaler.pkl` or `preprocessor.pkl` – final StandardScaler (and/or combined preprocessing dictionary) applied after concatenating Parselmouth and PCA‑reduced features.

You usually only re‑run this if you change the dataset or feature choices.

### 4.4 Train a model

There are two typical training scripts: a simple baseline and a tuned one.

#### Option A – baseline training

python train_model.py

This script:

- Loads `X_features_scaled.npy` and `y_labels.npy`.
- Splits into training and validation sets.
- Trains a baseline RandomForest (or similar).
- Prints metrics (accuracy, precision, recall, F1, confusion matrix).
- Saves the model as, for example:

speech_state_model.pkl


#### Option B – tuned model (recommended)

python train_model.py

These scripts typically:

- Use stratified cross‑validation.
- Run hyperparameter search (e.g. `RandomizedSearchCV` for RandomForest).
- Print:
  - Best hyperparameters.
  - Cross‑validated macro F1.
  - Validation classification report and confusion matrix.
- Save the tuned model as:

speech_state_model_tuned.pkl

You can inspect the printed metrics to confirm performance (around ~80%+ accuracy on RAVDESS‑based states is expected).

### 4.5 Use your own trained model in the frontend

To make the web app use your freshly trained model instead of the shipped one:

1. Ensure the following files are present in the project root:

   - `speech_state_model_tuned.pkl`  (or rename your own model file to this name).
   - `pca_os_features.pkl`
   - `scaler.pkl`

2. Start the app again:

python frontend.py

Now, all predictions come from **your** model trained on the latest dataset.

---

## 5. Notes and limitations

- The model is trained on RAVDESS, which contains acted English speech in a controlled environment.  
  Results may be less accurate on noisy or very different recordings.
- Predictions are coarse emotional “states” and should **not** be used for any clinical or high‑stakes decisions.
- Integration with IoT devices in the present scenario is a costly thing, as it requires individual students to have a separate microphone for the best results.
- The RAVDESS dataset is subject to its own license; this repository does not redistribute the audio. You must obtain it from official sources.

---

## 6. Future Scope

- This project is intended to be integrated with an IoT system, such as an audio recording device, which records and sends the audio clip of a student speaking during the classroom interaction **in real time**, to a processing node which contains this script.
- The audio file is preprocessed using **frontend.py** logic, exactly as in training phase.
- The model predicts the state of the speaker and shows the output in real time.
- This way, based on the output of the system, necessary steps can be suggested/taken to improve student interaction and to make teaching learning activities more effective.

## 7. Credits

- **Dataset:** RAVDESS – Ryerson Audio‑Visual Database of Emotional Speech and Song.
- **Feature extraction:** Praat (via Parselmouth) and openSMILE.
- **Web & ML stack:** Flask, scikit‑learn, numpy, pandas, etc.

## 8. Authors
 - **Suvash Sharma Subedi** (22BCP116)
 - **Satyam Mishra** (22BCP044)

Final year B.Tech CSE students in PDEU, Gujarat India.

Submitted to Dr. Tanmay Bhowmik as a Minor Project Report under the title: `Prosody-assisted device to monitor student engagement in smart classroom environment`