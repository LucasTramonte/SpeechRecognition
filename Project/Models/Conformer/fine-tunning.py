import os
import torch
import librosa
import random
import numpy as np
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass
from typing import Dict, List, Union

# Set seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Local dataset path
dataset_root = os.path.join("..", "..", "audio", "Datasets", "Fine-tunning", "LibriSpeech", "dev-clean")

# Model name
MODEL_NAME = "facebook/wav2vec2-conformer-rope-large-960h-ft"

# Define save path for fine-tuned model
save_dir = os.path.join(os.getcwd(), "fine-tunning", "cont-fnt-conf")  # Save inside "fine-tunning/"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

# Load pre-trained model & processor
try:
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCTC.from_pretrained(MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# Expecting structure: dataset_root/{speaker}/{chapter}/{file_id}.flac + dataset_root/{speaker}/{chapter}.trans.txt
audio_files = []
transcriptions = {}

# Load the evaluation dataset from your local "test-clean" dataset
test_root = os.path.join("..", "..", "audio", "Datasets", "Test", "LibriSpeech", "test-clean")

# Load evaluation dataset
test_files = []
test_transcriptions = {}

for speaker in os.listdir(test_root):
    speaker_path = os.path.join(test_root, speaker)
    if not os.path.isdir(speaker_path):
        continue

    for chapter in os.listdir(speaker_path):
        chapter_path = os.path.join(speaker_path, chapter)
        if not os.path.isdir(chapter_path):
            continue

        # Load test transcriptions
        transcript_file = os.path.join(chapter_path, f"{speaker}-{chapter}.trans.txt")
        if os.path.exists(transcript_file):
            with open(transcript_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        file_id, text = parts
                        test_transcriptions[file_id] = text

        # Load test audio files
        for file in os.listdir(chapter_path):
            if file.endswith(".flac"):
                file_path = os.path.join(chapter_path, file)
                file_id = file.replace(".flac", "")

                if file_id in test_transcriptions:
                    test_files.append((file_path, test_transcriptions[file_id]))


# Convert dataset into a format suitable for training
def prepare_dataset(audio_text_pair):
    audio_path, text = audio_text_pair
    speech, sr = librosa.load(audio_path, sr=16000)  # Load audio at 16kHz

    # Convert text to token IDs
    inputs = processor(speech, sampling_rate=sr, text=text, return_tensors="pt", padding=True)
    return {
        "input_values": inputs.input_values[0],
        "labels": inputs.labels[0]
    }

# Convert test dataset to model format
test_dataset = [prepare_dataset(pair) for pair in test_files]
print("test_dataset downloaded")


# Scan dataset directory
for speaker in os.listdir(dataset_root):
    speaker_path = os.path.join(dataset_root, speaker)
    if not os.path.isdir(speaker_path):
        continue

    for chapter in os.listdir(speaker_path):
        chapter_path = os.path.join(speaker_path, chapter)
        if not os.path.isdir(chapter_path):
            continue

        # Load transcriptions
        transcript_file = os.path.join(chapter_path, f"{speaker}-{chapter}.trans.txt")
        if os.path.exists(transcript_file):
            with open(transcript_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        file_id, text = parts
                        transcriptions[file_id] = text

        # Load audio files
        for file in os.listdir(chapter_path):
            if file.endswith(".flac"):
                file_path = os.path.join(chapter_path, file)
                file_id = file.replace(".flac", "")

                if file_id in transcriptions:
                    audio_files.append((file_path, transcriptions[file_id]))

# Convert dataset into a format suitable for training
def prepare_dataset(audio_text_pair):
    audio_path, text = audio_text_pair
    speech, sr = librosa.load(audio_path, sr=16000)  # Load audio at 16kHz

    # Convert text to token IDs
    inputs = processor(speech, sampling_rate=sr, text=text, return_tensors="pt", padding=True)
    return {
        "input_values": inputs.input_values[0],
        "labels": inputs.labels[0]
    }

# Process all data
dataset = [prepare_dataset(pair) for pair in audio_files]

# Define data collator to handle padding
@dataclass
class DataCollatorCTC:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Union[torch.Tensor, List[int]]]]) -> Dict[str, torch.Tensor]:
        input_values = [torch.tensor(feature["input_values"]) for feature in features]
        labels = [torch.tensor(feature["labels"]) for feature in features]

        batch = {
            "input_values": torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True),
        }
        return batch

data_collator = DataCollatorCTC(processor=processor)

# Training arguments (Save model in `fine-tunning/cont-fnt-conf`)
training_args = TrainingArguments(
    output_dir=save_dir,                # Save model inside "fine-tunning/cont-fnt-conf"
    per_device_train_batch_size=4,      # Adjust batch size for GPU memory
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,                  # Can be fine-tuned further
    warmup_steps=500,
    save_total_limit=2,
    gradient_checkpointing=True,         # Saves memory
    fp16=True if torch.cuda.is_available() else False,  # Use FP16 if available
    push_to_hub=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=test_dataset,  
    tokenizer=processor,
    data_collator=data_collator,
)

# Start fine-tuning
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model inside "fine-tunning/cont-fnt-conf"
trainer.save_model(save_dir)
processor.save_pretrained(save_dir)

print(f"Fine-tuning complete! Model saved in {save_dir}")
