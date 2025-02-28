import os
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from torch.nn.functional import cosine_similarity

# Load the Titanet model from NVIDIA NeMo
MODEL_NAME = "titanet_large"
model = EncDecSpeakerLabelModel.from_pretrained(model_name=MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to extract speaker embeddings (pass file path instead of tensor)
def get_embedding(audio_path):
    with torch.no_grad():
        embedding = model.get_embedding(audio_path)
    return embedding

# Set the root of the LibriSpeech dev-clean dataset
dataset_root = os.path.join("..", "audio", "Dataset", "Test", "LibiSpeech", "dev-clean")

# Create a dictionary mapping speaker IDs to a list of their audio file paths
speaker_files = {}
for speaker_id in os.listdir(dataset_root):
    speaker_dir = os.path.join(dataset_root, speaker_id)
    if os.path.isdir(speaker_dir):
        speaker_files[speaker_id] = []
        # Each speaker directory has multiple chapter directories
        for chapter in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter)
            if os.path.isdir(chapter_dir):
                # Look for .flac audio files in the chapter directory
                for file in os.listdir(chapter_dir):
                    if file.endswith(".flac"):
                        audio_file = os.path.join(chapter_dir, file)
                        speaker_files[speaker_id].append(audio_file)

# Now, for each speaker, extract embeddings for all their audio files
speaker_embeddings = {}
for speaker_id, files in speaker_files.items():
    embeddings = []
    print(f"Processing speaker {speaker_id} with {len(files)} files.")
    for file in files:
        embedding = get_embedding(file)
        embeddings.append(embedding)
    # Stack all embeddings for this speaker into a tensor
    speaker_embeddings[speaker_id] = torch.stack(embeddings)

# Optionally, compute an average embedding per speaker to represent that speaker overall
speaker_avg_embeddings = {}
for speaker_id, emb_tensor in speaker_embeddings.items():
    # Average over the first dimension (all files for that speaker)
    speaker_avg_embeddings[speaker_id] = emb_tensor.mean(dim=0)

# Now, compute the cosine similarity between speakers' average embeddings
speaker_ids = list(speaker_avg_embeddings.keys())
for i in range(len(speaker_ids)):
    for j in range(i + 1, len(speaker_ids)):
        emb1 = speaker_avg_embeddings[speaker_ids[i]].unsqueeze(0)
        emb2 = speaker_avg_embeddings[speaker_ids[j]].unsqueeze(0)
        sim = cosine_similarity(emb1, emb2)
        print(f"Similarity between speaker {speaker_ids[i]} and {speaker_ids[j]}: {sim.item():.4f}")
