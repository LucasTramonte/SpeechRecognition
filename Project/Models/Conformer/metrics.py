import pandas as pd
import jiwer
import os

# Load data from CSV
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "results", "transcriptions.csv")

df = pd.read_csv(csv_path)

global_wer = jiwer.wer(df["Ground_Truth"].tolist(), df["Transcription"].tolist())
print(f"Word Error Rate (WER): {global_wer:.2%}")

# Function to calculate WER and different words
def calculate_wer_and_diff(row):
    ground_truth = row["Ground_Truth"]
    transcription = row["Transcription"]
    
    # Calculate WER
    wer = jiwer.wer(ground_truth, transcription)
    
    # Split words to find differences
    truth_words = set(ground_truth.split())
    trans_words = set(transcription.split())
    
    # Find different words
    different_words = trans_words.symmetric_difference(truth_words)  # Words that are in one set but not in the other
    
    return pd.Series([wer, " ".join(different_words)])

# Apply the function to each row in the DataFrame
df[["WER", "Different_Words"]] = df.apply(calculate_wer_and_diff, axis=1)

# Display the results
print(df)

# Save the results to CSV
results_dir =  os.path.join(os.getcwd(), "results")
results_path = os.path.join(results_dir, "transcription_wer_final.csv")
df.to_csv(results_path, index=False)

print("Analysis saved in 'transcription_wer_final.csv'")
