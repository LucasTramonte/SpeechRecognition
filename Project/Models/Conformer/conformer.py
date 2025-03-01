import os
import torch
import librosa
import pandas as pd
from transformers import AutoProcessor, AutoModelForCTC

# Dataset path
dataset_root = os.path.join("..","..", "audio", "Datasets", "Test", "LibriSpeech", "test-clean")

# Load the model correctly
MODEL_NAME = "facebook/wav2vec2-conformer-rope-large-960h-ft"

try:
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCTC.from_pretrained(MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Using device: {device}")

# Store transcriptions
transcription_results = []

# Iterate through dataset directories
for speaker_id in sorted(os.listdir(dataset_root)):  # Sorting for better tracking
    speaker_dir = os.path.join(dataset_root, speaker_id)
    if os.path.isdir(speaker_dir):
        print(f"Processing speaker: {speaker_id}")
        
        for chapter_id in sorted(os.listdir(speaker_dir)):
            chapter_dir = os.path.join(speaker_dir, chapter_id)
            if os.path.isdir(chapter_dir):
                print(f"  Processing chapter: {chapter_id}")

                # Load correct transcriptions
                transcription_file = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}.trans.txt")
                ground_truth_dict = {}

                if os.path.exists(transcription_file):
                    with open(transcription_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split(" ", 1)
                            if len(parts) == 2:
                                file_id, text = parts
                                ground_truth_dict[file_id] = text

                # Process each audio file
                for file in sorted(os.listdir(chapter_dir)):  # Sorting for better tracking
                    if file.endswith(".flac"):
                        audio_path = os.path.join(chapter_dir, file)
                        file_id = file[:-5]  # Remove ".flac" extension

                        # print(f"    Processing file: {file}")

                        # Get the correct transcription
                        ground_truth = ground_truth_dict.get(file_id, "")

                        # Load audio
                        try:
                            speech, sr = librosa.load(audio_path, sr=16000)
                        except Exception as e:
                            print(f"    Error loading {audio_path}: {e}")
                            continue

                        # Convert audio to model input
                        try:
                            inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
                            inputs = inputs.to(device)  # Move to GPU
                        except Exception as e:
                            print(f"    Error processing audio: {e}")
                            continue

                        # Model inference
                        try:
                            with torch.no_grad():
                                logits = model(**inputs).logits
                        except Exception as e:
                            print(f"    Error during inference: {e}")
                            continue

                        # Decode transcription
                        try:
                            predicted_ids = torch.argmax(logits, dim=-1)
                            transcription = processor.batch_decode(predicted_ids)[0]
                        except Exception as e:
                            print(f"    Error decoding: {e}")
                            transcription = ""

                        # Store result
                        transcription_results.append({
                            "Speaker_ID": speaker_id,
                            "Chapter_ID": chapter_id,
                            "File_Path": audio_path,
                            "Transcription": transcription,
                            "Ground_Truth": ground_truth
                        })

# Save CSV
df = pd.DataFrame(transcription_results)
df.to_csv("conformer_transcriptions.csv", index=False)
print("Transcriptions saved in 'conformer_transcriptions.csv'.")