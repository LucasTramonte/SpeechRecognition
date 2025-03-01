import os
import torch
import librosa
import pandas as pd
import jiwer
from transformers import AutoProcessor, AutoModelForCTC

# Define dataset path
dataset_root = os.path.join("..", "..", "audio", "Datasets", "Test", "LibriSpeech", "test-clean", "121", "121726")

# Path to transcription file
transcription_file = os.path.join(dataset_root, "121-121726.trans.txt")

# Speech recognition model
MODEL_NAME = "facebook/wav2vec2-conformer-rope-large-960h-ft"

# Load model and processor
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCTC.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Configure GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load ground truth transcriptions
ground_truth_dict = {}
if os.path.exists(transcription_file):
    with open(transcription_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                file_id, text = parts
                ground_truth_dict[file_id] = text

# Process specific audio files
transcription_results = []
for i in range(15):  # From 0000 to 0014
    file_name = f"121-121726-{i:04d}.flac"
    audio_path = os.path.join(dataset_root, file_name)
    file_id = file_name[:-5]  # Remove ".flac" extension

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        continue

    # Get ground truth transcription
    ground_truth = ground_truth_dict.get(file_id, "")

    # Load audio
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        continue

    # Convert audio for model input
    try:
        inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = inputs.to(device)  # Move to GPU
    except Exception as e:
        print(f"Error processing audio: {e}")
        continue

    # Model inference
    try:
        with torch.no_grad():
            logits = model(**inputs).logits
    except Exception as e:
        print(f"Error during inference: {e}")
        continue

    # Decode transcription
    try:
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
    except Exception as e:
        print(f"Error decoding: {e}")
        transcription = ""

    # Store results
    transcription_results.append({
        "File_Name": file_name,
        "Transcription": transcription,
        "Ground_Truth": ground_truth
    })

# Create DataFrame to view results
df = pd.DataFrame(transcription_results)

# Save results to CSV in a results folder
results_dir =  os.path.join(os.getcwd(), "results")
results_path = os.path.join(results_dir, "transcription_test.csv")
df.to_csv(results_path, index=False)