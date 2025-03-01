import pandas as pd
import jiwer
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "results", "transcription_test.csv")

df = pd.read_csv(csv_path)

# Function to normalize text (remove punctuation, convert to uppercase)
def normalize_text(text):
    text = text.upper()  # Convert to uppercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Apply normalization to both columns
df["Ground_Truth"] = df["Ground_Truth"].apply(normalize_text)
df["Transcription"] = df["Transcription"].apply(normalize_text)

# Compute global WER
global_wer = jiwer.wer(df["Ground_Truth"].tolist(), df["Transcription"].tolist())
print(f"Word Error Rate (WER): {global_wer:.2%}")

# Function to calculate WER and find different words
def calculate_wer_and_diff(row):
    ground_truth = row["Ground_Truth"]
    transcription = row["Transcription"]
    
    # Calculate WER
    wer = jiwer.wer(ground_truth, transcription)
    
    # Split words to find differences
    truth_words = set(ground_truth.split())
    trans_words = set(transcription.split())
    
    # Find different words
    different_words = trans_words.symmetric_difference(truth_words)  # Words that are in one but not the other
    
    return pd.Series([wer, " ".join(different_words)])

# Apply the function to each row in the DataFrame
df[["WER", "Different_Words"]] = df.apply(calculate_wer_and_diff, axis=1)

# Display the results
print(df)

# Save the results to CSV
results_dir = os.path.join(os.getcwd(), "results")
results_path = os.path.join(results_dir, "transcription_wer.csv")
df.to_csv(results_path, index=False)

print(f"Analysis saved in '{results_path}'")
