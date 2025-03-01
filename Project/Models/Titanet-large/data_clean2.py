import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================
# Class: SpeakerSimilarityAnalyzer
# ===================================
class SpeakerSimilarityAnalyzer:
    def __init__(self, file_path):
        """Initialize the analyzer with a CSV file path."""
        self.file_path = file_path
        self.df = None
        self.summary_df = None

    def load_data(self):
        """Loads the CSV file containing speaker similarity data."""
        self.df = pd.read_csv(self.file_path)
        print(f" Data loaded successfully with {self.df.shape[0]} rows.")

    def plot_similarity_distribution(self):
        """Creates a boxplot to show the similarity distribution per speaker."""
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Speaker_1", y="Cosine_Similarity", data=self.df)
        plt.xticks(rotation=90)
        plt.title("Speaker Similarity Distribution")
        plt.xlabel("Speaker")
        plt.ylabel("Cosine Similarity")
        plt.show()
        print("Similarity distribution plotted.")

    def plot_similarity_heatmap(self):
        """Creates a heatmap to visualize speaker similarity relationships."""
        pivot_df = self.df.pivot(index="Speaker_1", columns="Speaker_2", values="Cosine_Similarity")

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, cmap="coolwarm", center=0, annot=False)
        plt.title("Speaker Similarity Heatmap")
        plt.xlabel("Speaker 2")
        plt.ylabel("Speaker 1")
        plt.show()
        print("Similarity heatmap plotted.")

    def run_analysis(self):
        """Runs the full analysis: Load, Compute Summary, and Generate Plots."""
        self.load_data()
        self.plot_similarity_distribution()
        self.plot_similarity_heatmap()
        print("Analysis completed successfully!")

# ===================================
# Run Analysis
# ===================================
if __name__ == "__main__":
    # 🔹 Replace 'speaker_similarity.csv' with the actual CSV file path
    csv_path = "speaker_similarity.csv"
    
    # Initialize the analyzer
    analyzer = SpeakerSimilarityAnalyzer(csv_path)
    
    # Run the full analysis
    analyzer.run_analysis()
