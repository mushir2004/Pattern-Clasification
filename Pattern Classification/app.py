from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import librosa

app = Flask(__name__)

# Load the trained GMM model
with open("gmm_models.pkl", "rb") as model_file:
    gmm_models = pickle.load(model_file)

emotion_labels = list(gmm_models.keys())  # Extract available emotions

def extract_features(file_path, n_mfcc=13):
    """Extract MFCC features from an audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)  # Take mean over time axis
    except Exception as e:
        return None

@app.route("/")
def home():
    return render_template("index.html")  # Serve frontend

@app.route("/predict", methods=["POST"])
def predict():
    """Handle audio file upload and predict emotion."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."})

    file = request.files["file"]
    file_path = "uploaded_audio.wav"
    file.save(file_path)

    # Extract features
    features = extract_features(file_path)
    if features is None:
        return jsonify({"error": "Feature extraction failed."})

    # Reshape for prediction
    features = features.reshape(1, -1)

    # Get prediction from GMM models
    scores = {emotion: gmm.score(features) for emotion, gmm in gmm_models.items()}
    predicted_emotion = max(scores, key=scores.get)

    return jsonify({"prediction": predicted_emotion})

if __name__ == "__main__":
    app.run(debug=True)
