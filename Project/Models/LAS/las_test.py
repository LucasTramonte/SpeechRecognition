import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import pandas as pd
import jiwer
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset path
dataset_root = os.path.join("..", "..", "audio", "Datasets", "Test", "LibriSpeech", "test-clean", "121", "121726")
transcription_file = os.path.join(dataset_root, "121-121726.trans.txt")

# Load ground truth transcriptions
ground_truth_dict = {}
if os.path.exists(transcription_file):
    with open(transcription_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                file_id, text = parts
                ground_truth_dict[file_id] = text

# Convert text to index sequences
char_vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")  # Character set including space and apostrophe
char_to_index = {c: i + 1 for i, c in enumerate(char_vocab)}  # Map each character to a unique index
char_to_index["<EOS>"] = len(char_vocab) + 1  # Add End-of-Sequence token

# Convert text to numerical sequence
def text_to_sequence(text):
    return [char_to_index[c] for c in text if c in char_to_index] + [char_to_index["<EOS>"]]

# Define LAS Model
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell  # Return encoder outputs and last hidden state

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, encoder_outputs):
        attn_weights = torch.tanh(self.attn(encoder_outputs))
        attn_scores = torch.matmul(attn_weights, self.v)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights.unsqueeze(2) * encoder_outputs, dim=1)
        return context

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_outputs):
        context = Attention(encoder_outputs.shape[-1])(encoder_outputs)
        lstm_out, _ = self.lstm(context.unsqueeze(1))
        output = self.fc(lstm_out.squeeze(1))
        return output

class LAS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LAS, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(output_dim, hidden_dim)

    def forward(self, x):
        encoder_outputs, _, _ = self.encoder(x)
        output = self.decoder(encoder_outputs)
        return output

# Initialize LAS model
input_dim = 40  # Mel Spectrogram Features
hidden_dim = 256
output_dim = len(char_to_index) + 1  # Number of output characters
model = LAS(input_dim, hidden_dim, output_dim).to(device)
model.eval()

# Process audio files
transcription_results = []
for i in range(15):  # Process files from 0000 to 0014
    file_name = f"121-121726-{i:04d}.flac"
    audio_path = os.path.join(dataset_root, file_name)
    file_id = file_name[:-5]

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        continue

    ground_truth = ground_truth_dict.get(file_id, "")

    # Load audio
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=speech, sr=sr, n_mels=40)
        mel_spec = torch.tensor(mel_spec.T).float().unsqueeze(0).to(device)  # Shape: (1, Time, Features)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        continue

    # Model inference
    try:
        with torch.no_grad():
            output = model(mel_spec)
            predicted_sequence = torch.argmax(output, dim=-1).squeeze().cpu().numpy()
            predicted_text = "".join([char_vocab[i - 1] for i in predicted_sequence if i > 0])
    except Exception as e:
        print(f"Error during inference: {e}")
        predicted_text = ""

    # Store results
    transcription_results.append({
        "File_Name": file_name,
        "Transcription": predicted_text,
        "Ground_Truth": ground_truth
    })

# Save results to CSV
df = pd.DataFrame(transcription_results)

# Define the results directory
results_dir =  os.path.join(os.getcwd(), "results")

# Save the transcription results in the "results" folder
results_path = os.path.join(results_dir, "transcription_test_results.csv")
df.to_csv(results_path, index=False)

# Calculate WER
wer_score = jiwer.wer(df["Ground_Truth"].tolist(), df["Transcription"].tolist())
print(f"LAS Word Error Rate (WER): {wer_score:.2%}")
