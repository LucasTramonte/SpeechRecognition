import os
import torch
import librosa
import pandas as pd
import whisper

# Root directory for the dataset
dataset_root = os.path.join("..", "..", "audio", "Datasets", "Test", "LibriSpeech", "test-clean")

# Load the Whisper model
# You can choose between "tiny", "base", "small", "medium", "large"
model = whisper.load_model("tiny")

# Define whether to use GPU (only if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# List to store transcription results
transcription_results = []

# Loop through all speaker directories in the dataset root
for speaker_dir in os.listdir(dataset_root):
    speaker_path = os.path.join(dataset_root, speaker_dir)
    if not os.path.isdir(speaker_path):
        continue

    # Loop through all chapter directories for the current speaker
    for chapter_dir in os.listdir(speaker_path):
        chapter_path = os.path.join(speaker_path, chapter_dir)
        if not os.path.isdir(chapter_path):
            continue

        # Path to the transcription file
        transcription_file = os.path.join(chapter_path, f"{speaker_dir}-{chapter_dir}.trans.txt")

        # Dictionary to store ground-truth transcriptions
        ground_truth_dict = {}
        if os.path.exists(transcription_file):
            with open(transcription_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        file_id, text = parts
                        ground_truth_dict[file_id] = text
        else:
            print(f"Transcription file not found: {transcription_file}")
            continue

        # Loop through audio files in the chapter directory
        for audio_file in os.listdir(chapter_path):
            if not audio_file.endswith(".flac"):
                continue

            audio_path = os.path.join(chapter_path, audio_file)
            file_id = audio_file[:-5]  # Remove ".flac" extension

            # Retrieve ground truth transcription (if available)
            ground_truth = ground_truth_dict.get(file_id, "")

            # Load the audio file with librosa
            try:
                speech, sr = librosa.load(audio_path, sr=16000)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue

            # Transcribe the audio with Whisper
            try:
                result = model.transcribe(speech, fp16=False)
                transcription = result["text"].strip()
            except Exception as e:
                print(f"Error transcribing {audio_path}: {e}")
                transcription = ""

            # Store transcription results
            transcription_results.append({
                "File_Name": audio_file,
                "Transcription": transcription,
                "Ground_Truth": ground_truth
            })

# Create a DataFrame for visualization
df = pd.DataFrame(transcription_results)

# Save results
results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "transcription_test.csv")
df.to_csv(results_path, index=False)

print(f"Results saved to: {results_path}")