import os
import json
import torch
import soundfile as sf
from nemo.collections.asr.models import EncDecSpeakerLabelModel

# ========== STEP 1: PREPARE DATA ==========
dataset_root = os.path.join("..", "audio", "Datasets", "Train", "LibriSpeech", "dev-clean")
manifest_path = "train_manifest.json"

manifest_data = []
for speaker_id in os.listdir(dataset_root):
    speaker_dir = os.path.join(dataset_root, speaker_id)
    if os.path.isdir(speaker_dir):
        for chapter in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter)
            if os.path.isdir(chapter_dir):
                for file in os.listdir(chapter_dir):
                    if file.endswith(".flac"):
                        audio_path = os.path.join(chapter_dir, file)
                        duration = sf.info(audio_path).duration
                        manifest_data.append({
                            "audio_filepath": audio_path,
                            "duration": duration,
                            "label": speaker_id
                        })

# Save the manifest file
with open(manifest_path, "w") as f:
    for entry in manifest_data:
        f.write(json.dumps(entry) + "\n")

print(f"Saved manifest file to {manifest_path} with {len(manifest_data)} samples.")

# ========== STEP 2: LOAD MODEL ==========
MODEL_NAME = "titanet_large"
model = EncDecSpeakerLabelModel.from_pretrained(model_name=MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ========== STEP 3: SET TRAINING DATA ==========
"""
model.setup_training_data(
    train_data_config={
        "manifest_filepath": manifest_path,
        "batch_size": 32,  # Adjust batch size based on GPU capacity
        "shuffle": True,  # Shuffle data for better generalization
        "num_workers": 4,  # Use multiple processes to load data
        "pin_memory": True,  # Optimizes data transfer to GPU
    }
)
"""
model.setup_training_data(train_data_config={"manifest_filepath": manifest_path, "batch_size": 32})

# ========== STEP 4: SET TRAINING PARAMETERS ==========
model.setup_optimization(
    optim_config={
        "lr": 1e-4,  # Learning rate
        "betas": (0.9, 0.98),  # Adam optimizer parameters
        "weight_decay": 1e-5,  # Prevents overfitting
        "sched": {"name": "CosineAnnealing", "min_lr": 1e-6},  # Learning rate schedule
    }
)

# ========== STEP 5: TRAIN ==========
print("\nStarting fine-tuning...")
model.fit(max_epochs=10)  # Adjust epochs based on dataset size

# ========== STEP 6: SAVE MODEL ==========
model.save_to("titanet_finetuned.nemo")

print("\nTraining complete! Model saved as titanet_finetuned.nemo.")
