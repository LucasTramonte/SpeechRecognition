import os
import pandas as pd
import difflib

# File paths
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
file1= os.path.join(SCRIPT_DIR, "results", "fine-transcription_test.csv")
file2= os.path.join(SCRIPT_DIR, "results", "transcription_test.csv")

# Load CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Ensure both files have the same structure
assert set(df1.columns) == set(df2.columns), "The CSV files have different columns!"

# Merge both DataFrames based on 'File_Name'
merged_df = df1.merge(df2, on="File_Name", suffixes=("_fine", "_original"))

# Filter only rows where transcriptions are different
diff_df = merged_df[merged_df["Transcription_fine"] != merged_df["Transcription_original"]]

# Function to highlight differences between transcriptions
def highlight_differences(text1, text2):
    diff = list(difflib.ndiff(text1.split(), text2.split()))
    highlighted_diff = []
    
    for word in diff:
        if word.startswith("-"):
            highlighted_diff.append(f"[REMOVED: {word[2:]}]")
        elif word.startswith("+"):
            highlighted_diff.append(f"[ADDED: {word[2:]}]")
        else:
            highlighted_diff.append(word[2:])
    
    return " ".join(highlighted_diff)

# Create a new column with highlighted differences
diff_df["Differences"] = diff_df.apply(lambda row: highlight_differences(row["Transcription_fine"], row["Transcription_original"]), axis=1)

# Save differences to a CSV file
results_dir =  os.path.join(os.getcwd(), "results")
results_path = os.path.join(results_dir, "transcription_differences.csv")

diff_df.to_csv(results_path, index=False, encoding="utf-8")
print(f"\nFound {len(diff_df)} differences between transcriptions.\n")

