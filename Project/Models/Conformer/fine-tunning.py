import os
import torch
import librosa
import random
import numpy as np
import tensorflow as tf
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass
from typing import Dict, List, Union

# Disable TensorFlow OneDNN Warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Fix TensorFlow deprecation warning for losses
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Choose one speaker from each dataset
selected_speaker_dev = "84"  
selected_speaker_test = "61"  

# Define dataset paths for selected speakers
dataset_root = os.path.join("..", "..", "audio", "Datasets", "Fine-tunning", "LibriSpeech", "dev-clean", selected_speaker_dev)
test_root = os.path.join("..", "..", "audio", "Datasets", "Test", "LibriSpeech", "test-clean", selected_speaker_test)

# Define model name and save path
MODEL_NAME = "facebook/wav2vec2-conformer-rope-large-960h-ft"
save_dir = os.path.join(os.getcwd(), "fine-tunning", "cont-fnt-conf")
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

# Function to load data for a selected speaker
def load_speaker_dataset(root_path, speaker_id):
    audio_files = []
    transcriptions = {}

    for chapter in os.listdir(root_path):
        chapter_path = os.path.join(root_path, chapter)
        if not os.path.isdir(chapter_path):
            continue

        transcript_file = os.path.join(chapter_path, f"{speaker_id}-{chapter}.trans.txt")
        if os.path.exists(transcript_file):
            with open(transcript_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        file_id, text = parts
                        transcriptions[file_id] = text

        for file in os.listdir(chapter_path):
            if file.endswith(".flac"):
                file_path = os.path.join(chapter_path, file)
                file_id = file.replace(".flac", "")

                if file_id in transcriptions:
                    audio_files.append((file_path, transcriptions[file_id]))

    return audio_files

# Load training and evaluation datasets (only one speaker)
train_files = load_speaker_dataset(dataset_root, selected_speaker_dev)
test_files = load_speaker_dataset(test_root, selected_speaker_test)

print(f"Loaded {len(train_files)} training files from speaker {selected_speaker_dev}")
print(f"Loaded {len(test_files)} evaluation files from speaker {selected_speaker_test}")

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

# Convert data to model format
train_dataset = [prepare_dataset(pair) for pair in train_files]
test_dataset = [prepare_dataset(pair) for pair in test_files]

# Define data collator to handle padding
@dataclass
class DataCollatorCTC:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Union[torch.Tensor, List[int]]]]) -> Dict[str, torch.Tensor]:
        input_values = [feature["input_values"].clone().detach() for feature in features]
        labels = [feature["labels"].clone().detach() for feature in features]

        batch = {
            "input_values": torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True),
        }
        return batch

data_collator = DataCollatorCTC(processor=processor)

# Training arguments (Fixed evaluation strategy and tokenizer deprecation)
training_args = TrainingArguments(
    output_dir=save_dir,                # Save model inside "fine-tunning/cont-fnt-conf"
    per_device_train_batch_size=4,      # Adjust batch size for GPU memory
    gradient_accumulation_steps=2,
    eval_strategy="steps",              # Fix deprecation warning (was `evaluation_strategy`)
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
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
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    preprocessors=processor,  # Fix tokenizer deprecation warning (was `tokenizer`)
    data_collator=data_collator,
)

# Start fine-tuning
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model inside "fine-tunning/cont-fnt-conf"
trainer.save_model(save_dir)
processor.save_pretrained(save_dir)

print(f"Fine-tuning complete! Model saved in {save_dir}")
