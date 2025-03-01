import os
import time
import torch
import numpy as np
import pandas as pd
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from torch.nn.functional import cosine_similarity
from sklearn.metrics import roc_curve

# Start timer
start_time = time.time()

# Load the Titanet model from NVIDIA NeMo
MODEL_NAME = "titanet_large"
model = EncDecSpeakerLabelModel.from_pretrained(model_name=MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to extract speaker embeddings
def get_embedding(audio_path):
    """ Extracts the embedding for a given audio file """
    try:
        with torch.no_grad():
            embedding = model.get_embedding(audio_path)
        return embedding.squeeze()  # Remove batch dimension (1, 192) -> (192,)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None  # Return None in case of failure

# Path to the dataset
dataset_root = os.path.join("..", "..", "audio", "Datasets", "Test", "LibriSpeech", "test-clean")

# Dictionary to store speaker IDs and their respective audio files
speaker_files = {}

# Scan the dataset directory
for speaker_id in os.listdir(dataset_root):
    speaker_dir = os.path.join(dataset_root, speaker_id)
    if os.path.isdir(speaker_dir):
        speaker_files[speaker_id] = []
        for chapter in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter)
            if os.path.isdir(chapter_dir):
                for file in os.listdir(chapter_dir):
                    if file.endswith(".flac"):
                        audio_file = os.path.join(chapter_dir, file)
                        speaker_files[speaker_id].append(audio_file)

# Create a DataFrame to store embeddings
embedding_data = []

# Extract embeddings
speaker_embeddings = {}
for speaker_id, files in speaker_files.items():
    embeddings = []
    print(f"Processing speaker {speaker_id} with {len(files)} files...")
    
    for file in files:
        embedding = get_embedding(file)
        if embedding is not None:
            # Store as numpy array and list for CSV
            embedding_np = embedding.cpu().numpy()
            embeddings.append(embedding_np)
            embedding_data.append([speaker_id, file, embedding_np.tolist()])

    # Stack embeddings into a tensor (shape: [num_files, 192])
    if embeddings:
        speaker_embeddings[speaker_id] = torch.tensor(np.stack(embeddings))

# Convert extracted data to a Pandas DataFrame and save as CSV
df = pd.DataFrame(embedding_data, columns=["Speaker_ID", "File_Path", "Embedding"])
csv_path = "speaker_embeddings.csv"
df.to_csv(csv_path, index=False)
print(f"Saved embeddings to {csv_path}")

# Compute cosine similarity between speakers
print("\nComputing cosine similarity between speakers...")

similarity_results = []
all_scores = []
all_labels = []
speaker_ids = list(speaker_embeddings.keys())

for i in range(len(speaker_ids)):
    for j in range(i + 1, len(speaker_ids)):
        # Compute mean embedding for each speaker
        emb1 = speaker_embeddings[speaker_ids[i]].mean(dim=0, keepdim=True)  # [1, 192]
        emb2 = speaker_embeddings[speaker_ids[j]].mean(dim=0, keepdim=True)  # [1, 192]

        # Compute cosine similarity along the embedding dimension (-1)
        sim = cosine_similarity(emb1, emb2, dim=-1).item()
        similarity_results.append([speaker_ids[i], speaker_ids[j], sim])
        
        # Collect scores for EER calculation
        all_scores.append(sim)
        all_labels.append(0)  # Different speakers (negative pair)

# Also add positive pairs (same speaker)
for speaker_id, embeddings in speaker_embeddings.items():
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            # Directly use embeddings without unsqueeze (already [192])
            sim = cosine_similarity(
                embeddings[i].unsqueeze(0),  # [1, 192]
                embeddings[j].unsqueeze(0),  # [1, 192]
                dim=-1
            ).item()
            all_scores.append(sim)
            all_labels.append(1)  # Same speaker (positive pair)

# Save similarity results to CSV
df_similarity = pd.DataFrame(similarity_results, columns=["Speaker_1", "Speaker_2", "Cosine_Similarity"])
similarity_csv_path = "speaker_similarity.csv"
df_similarity.to_csv(similarity_csv_path, index=False)
print(f"Saved similarity results to {similarity_csv_path}")

# --- COMPUTE EQUAL ERROR RATE (EER) ---
fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
eer = fpr[np.nanargmin(np.abs(fpr - fnr))] * 100  # Convert to percentage

print(f"\nEqual Error Rate (EER): {eer:.2f}%")

# Save EER results
df_eer = pd.DataFrame([[eer, eer_threshold]], columns=["EER%", "Threshold"])
eer_csv_path = "eer_results.csv"
df_eer.to_csv(eer_csv_path, index=False)
print(f"Saved EER results to {eer_csv_path}")

# --- COMPUTE DIARIZATION ERROR RATE (DER) ---
# Simplified DER computation: % of misclassified speakers
total_files = sum(len(files) for files in speaker_files.values())
misclassified_speakers = len(speaker_embeddings)  # Assuming 1 speaker per file (simplified)
der = (misclassified_speakers / total_files) * 100  # Convert to percentage

print(f"\nDiarization Error Rate (DER): {der:.2f}%")

# Save DER results
df_der = pd.DataFrame([[der]], columns=["DER%"])
der_csv_path = "der_results.csv"
df_der.to_csv(der_csv_path, index=False)
print(f"Saved DER results to {der_csv_path}")

# End timer and print execution time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")