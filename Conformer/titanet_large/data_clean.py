import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import sys

# ===================================
# Load and Organize Data
# ===================================
def load_and_process_csv(file_path):
    """Loads the CSV file and processes embeddings into numpy arrays."""
    df = pd.read_csv(file_path)

    # Convert embeddings from string to numpy array
    df['Embedding'] = df['Embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

    return df

# ===================================
# Generate Summary Table
# ===================================
def create_summary_table(df):
    """Generates a summary table including embedding dimensions per speaker."""
    
    def get_embedding_shape(embeddings):
        """Extracts the embedding dimensions"""
        shapes = [e.shape for e in embeddings]
        unique_shapes = set(shapes)
        return unique_shapes  # Returns all unique embedding shapes

    summary_df = df.groupby("Speaker_ID").agg(
        Num_Files=("File_Path", "count"),
        Embedding_Shapes=("Embedding", lambda x: get_embedding_shape(x))
    ).reset_index()

    # Display summary table
    print(summary_df)  # Print in terminal
    summary_df.to_csv("speaker_summary.csv", index=False)  # Save as CSV

# ===================================
# Visualizing Embeddings
# ===================================
def plot_embeddings(df, method="tsne"):
    """Reduces embedding dimensions and plots them in 2D."""
    embeddings = np.vstack(df['Embedding'].values)
    speaker_ids = df['Speaker_ID'].values

    # Dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2)
        title = "PCA Visualization of Speaker Embeddings"
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        title = "t-SNE Visualization of Speaker Embeddings"

    reduced_embeddings = reducer.fit_transform(embeddings)

    # Create DataFrame for visualization
    plot_df = pd.DataFrame(reduced_embeddings, columns=["Dim1", "Dim2"])
    plot_df["Speaker_ID"] = speaker_ids

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Dim1", y="Dim2", hue="Speaker_ID", palette="tab10", data=plot_df, s=70, alpha=0.7
    )
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Speaker ID", bbox_to_anchor=(1, 1))
    plt.show()

# ===================================
# Run Analysis
# ===================================
if __name__ == "__main__":
    csv_path = "speaker_embeddings.csv"
    df = load_and_process_csv(csv_path)

    # Display Summary Table
    create_summary_table(df)

    # Plot Embeddings (choose 'tsne' or 'pca')
    plot_embeddings(df, method="tsne")

    sys.exit()