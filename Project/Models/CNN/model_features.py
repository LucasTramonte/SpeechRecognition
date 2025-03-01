import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import tensorflow as tf

Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
to_categorical = tf.keras.utils.to_categorical
save_model = tf.keras.models.save_model
load_model = tf.keras.models.load_model

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model, load_model

"""

# ===================================
# Audio Processing (Fixed-Length)
# ===================================
class AudioProcessor:
    def __init__(self, sample_rate=16000, duration=3):
        self.sample_rate = sample_rate
        self.duration = duration

    def load_audio(self, file_path):
        """Load and pad/trim audio to fixed duration."""
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        y, _ = librosa.effects.trim(y)  # Trim silence
        
        # Pad or truncate to fixed duration
        target_length = self.sample_rate * self.duration
        if len(y) > target_length:
            y = y[:target_length]
        else:
            padding = target_length - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        
        return y, sr

# ===================================
# Feature Extraction (MFCC Matrix)
# ===================================
class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_features(self, y):
        """Extract full MFCC matrix."""
        mfccs = librosa.feature.mfcc(
            y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        return mfccs

# ===================================
# CNN Speaker Classification Model
# ===================================
class SpeakerClassifier:
    def __init__(self, input_shape, num_classes):
        self.model = self._build_model(input_shape, num_classes)

    def _build_model(self, input_shape, num_classes):
        """Construct a CNN model."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )
        return history

    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"🔹 Model Accuracy: {acc:.2f}")
        
        # Confusion Matrix
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        return acc

    def save_model(self, filename="speaker_cnn.h5"):
        save_model(self.model, filename)
        print(f"Model saved as {filename}")

    def load_model(self, filename="speaker_cnn.h5"):
        self.model = load_model(filename)
        print(f"Model loaded from {filename}")

# ===================================
# Main Pipeline
# ===================================
if __name__ == "__main__":
    audio_folder = "audios/"  # Update with your path
    speakers = {"speaker1":0, "speaker2":1, "speaker3":2, "speaker4":3, "speaker5":4}

    # Initialize components
    processor = AudioProcessor(sample_rate=16000, duration=3)
    extractor = FeatureExtractor(sample_rate=16000, n_mfcc=13)
    data = []
    labels = []

    # Process audio files
    for speaker, label in speakers.items():
        speaker_path = os.path.join(audio_folder, speaker)
        if not os.path.exists(speaker_path):
            continue
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                y, sr = processor.load_audio(os.path.join(speaker_path, file))
                mfccs = extractor.extract_features(y)
                # Reshape for CNN (add channel dimension)
                mfccs = mfccs[..., np.newaxis]  # Shape: (n_mfcc, time_steps, 1)
                data.append(mfccs)
                labels.append(label)

    # Convert to arrays
    X = np.array(data)
    y = to_categorical(np.array(labels))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train CNN
    input_shape = (extractor.n_mfcc, X_train.shape[2], 1)  # (13, time_steps, 1)
    classifier = SpeakerClassifier(input_shape, len(speakers))
    classifier.train(X_train, y_train, X_test, y_test, epochs=20)
    classifier.evaluate(X_test, y_test)
    classifier.save_model()