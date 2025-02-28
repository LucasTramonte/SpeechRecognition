import torch
import soundfile as sf
import os
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from torch.nn.functional import cosine_similarity
import nemo.collections.asr as nemo_asr

# Load the Titanet model from NVIDIA NeMo
MODEL_NAME = "titanet_large"
model = EncDecSpeakerLabelModel.from_pretrained(model_name=MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to extract speaker embeddings (pass file path instead of tensor)
def get_embedding(audio_path):
    # Titanet expects the path to the file, not a tensor
    with torch.no_grad():
        embeddings = model.get_embedding(audio_path)  # Pass the file path

    return embeddings

# Paths to audio files
audio_path1 = os.path.join("..", "audio", "exemple", "an255-fash-b.wav")
audio_path2 = os.path.join("..", "audio","exemple", "cen7-fash-b.wav")

# Extract embeddings
emb1 = get_embedding(audio_path1)
emb2 = get_embedding(audio_path2)

# Compute cosine similarity between embeddings
similarity = cosine_similarity(emb1, emb2)

print(f"Similarity between speakers: {similarity.item():.4f}")

# speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
# emb = speaker_model.get_embedding(audio_path1)
# speaker_model.verify_speakers(audio_path1, audio_path2)
