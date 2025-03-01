import os
import torch
import librosa
import pandas as pd
import whisper

# Defina o caminho exato (igual ao seu script wav2vec2)
dataset_root = os.path.join("..", "..", "audio", "Datasets", "Test", "LibriSpeech", "test-clean", "121", "121726")

# Caminho para o arquivo de transcrição
transcription_file = os.path.join(dataset_root, "121-121726.trans.txt")

# Carrega modelo do Whisper
# Pode escolher "tiny", "base", "small", "medium", "large" etc.
# Se estiver apenas com CPU, use modelos menores para ter melhor velocidade.
model = whisper.load_model("tiny")  

# Define se vai usar GPU (apenas se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dicionário para transcrições de referência
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

transcription_results = []
for i in range(15):  # de 0000 a 0014
    file_name = f"121-121726-{i:04d}.flac"
    audio_path = os.path.join(dataset_root, file_name)
    file_id = file_name[:-5]  # remove ".flac"
    
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        continue

    ground_truth = ground_truth_dict.get(file_id, "")

    # Carrega o áudio com librosa (opcional — você também pode passar só o path pro Whisper)
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        continue
    
    # Transcreve com Whisper
    # Se você estiver em CPU, pode usar: model.transcribe(speech, fp16=False)
    # Passamos diretamente o array 'speech' e sample_rate=16000
    try:
        result = model.transcribe(speech, fp16=False)  # Força FP32 no CPU
        transcription = result["text"].strip()
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        transcription = ""

    # Guarda resultados
    transcription_results.append({
        "File_Name": file_name,
        "Transcription": transcription,
        "Ground_Truth": ground_truth
    })

# Cria DataFrame para visualizar
df = pd.DataFrame(transcription_results)

# Salva resultados
results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "transcription_test.csv")
df.to_csv(results_path, index=False)

print(f"Results saved to: {results_path}")
