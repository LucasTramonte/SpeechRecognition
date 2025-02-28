import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import joblib  # To save the model

# ===================================
# Audio Processing
# ===================================
class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def load_audio(self, file_path):
        """Loads an audio file and normalizes it to a fixed sampling rate."""
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        y, _ = librosa.effects.trim(y)  # Removes leading and trailing silence
        return y, sr

    def plot_spectrogram(self, y, sr):
        """Displays the spectrogram of the audio."""
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.show()

# ===================================
# Feature Extraction
# ===================================
class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def extract_features(self, y):
        """Extracts MFCCs, spectral frequency, and statistical features from the audio."""
        features = {}

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        for i in range(self.n_mfcc):
            features[f'MFCC_{i+1}'] = np.mean(mfccs[i])

        # Spectral Features
        features['Zero_Crossing_Rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        features['Spectral_Centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sample_rate))
        features['Spectral_Bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate))
        features['RMS_Energy'] = np.mean(librosa.feature.rms(y=y))

        return features

# ===================================
#  Speaker Classification Model
# ===================================
class SpeakerClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X, y):
        """Trains the speaker classification model."""
        self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        """Evaluates the model and prints performance metrics."""
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"🔹 Model Accuracy: {acc:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def save_model(self, filename="speaker_model.pkl"):
        """Saves the trained model."""
        joblib.dump(self.model, filename)
        print(f" Model saved as {filename}")

    def load_model(self, filename="speaker_model.pkl"):
        """Loads a trained model."""
        self.model = joblib.load(filename)
        print(f" Model loaded from {filename}")

# ===================================
#  Main Pipeline: Running the Entire Process
# ===================================
if __name__ == "__main__":
    # Audio directory
    audio_folder = "audios/"  # Replace with your actual audio directory
    speakers = {
        "speaker1": 0, "speaker2": 1, "speaker3": 2, "speaker4": 3, "speaker5": 4
    }  # Speaker-to-label mapping

    # Initialize classes
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    classifier = SpeakerClassifier()

    # Extracting features from audio files
    data = []
    labels = []

    for speaker, label in speakers.items():
        speaker_path = os.path.join(audio_folder, speaker)
        if not os.path.exists(speaker_path):
            continue

        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_path, file)

                # Process audio
                y, sr = processor.load_audio(file_path)

                # Extract features
                features = extractor.extract_features(y)
                data.append(features)
                labels.append(label)

    # 🔹 Convert to DataFrame
    df = pd.DataFrame(data)
    df['Label'] = labels
    df.to_csv("features.csv", index=False)  # Save features for analysis

    # 🔹 Prepare data for training
    X = df.drop(columns=['Label']).values
    y = df['Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔹 Train and evaluate the model
    classifier.train(X_train, y_train)
    classifier.evaluate(X_test, y_test)

    # 🔹 Save the trained model
    classifier.save_model()
