import pandas as pd
import jiwer

# Load data from CSV
df = pd.read_csv("transcription_results.csv")
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
df.to_csv("transcription_wer.csv", index=False)
print("Analysis saved in 'transcription_wer.csv'")
