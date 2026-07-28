import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# data = np.load("embeddings/all/BGE_Large_embeddings_pca_full.npz")

# Prints: 'embeddings'
# print(data.files)

# Prints: (2, 1024)
# print(data["embeddings"].shape) 

def plot_pca(pca_file, title):
    """
    Plot PCA embeddings from a .npz file.

    Args:
        pca_file (str): Path to the .npz file containing PCA embeddings.

    Returns:
        None
    """
    data = np.load(pca_file)
    embeddings = data["embeddings"]
    embeddings_transformed = embeddings.T

    # Check if embeddings have at least 2 dimensions
    if embeddings_transformed.shape[1] < 2:
        raise ValueError("Embeddings must have at least 2 dimensions for plotting.")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_transformed[:, 0], embeddings_transformed[:, 1], alpha=0.5)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    # plt.title((Path(pca_file).stem).replace("_", " "))
    if title is None:
        plt.title("PCA Embeddings")
    else:
        plt.title(title)

    output_filename = f"{Path(pca_file).parent}/{Path(pca_file).stem}_plot.png"
    print("Saving plot to:", output_filename)
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()