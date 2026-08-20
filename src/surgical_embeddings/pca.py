import numpy as np
from importlib.resources import files

from pathlib import Path

# Constants
PCA_FILES = {
    "SapBERT": "sapbert_full_pca_full.npz",
    "MiniLM": "minilm_pca_full.npz",
    "BGE_Large": "bge_large_pca_full.npz",
}

# Full PCA function
def apply_full_pca(X, model_name, variance_percent=None):
    """
    Apply precomputed PCA transformation.

    Args:
        X (np.ndarray): Original embeddings.
        model_name (str): Model used to generate embeddings.
        variance_percent (float | None): Percentage of variance to retain. If None, return full PCA representation.

        Example:
            95 -> keep enough PCs for 95% variance

    Returns: 
        np.ndarray: PCA transformed embeddings
    """

    pca = load_pca_model(model_name)

    # Validate embedding dimension
    if X.shape[1] != pca["n_features"]:
        raise ValueError(
            f"Embedding dimension mismatch. Received {X.shape[1]}, expected {pca['n_features']}.")

    # Full PCA projection
    X_pca = (X - pca["mean"]) @ pca["components"].T

    # Full PCA requested
    if variance_percent is None:
        return X_pca

    # Determine number of components
    k = get_num_components(pca["explained_variance_ratio"], variance_percent)
    return X_pca[:, :k]

# Load PCA matrix file
def load_pca_model(model_name):
    """
    Load PCA matrix file.

    Args:
        model_name (str): Model used to generate embeddings.

    Returns:
        dict: PCA parameters including mean, components, and explained variance ratio.
    """

    if model_name not in PCA_FILES:
        raise ValueError(f"No PCA file available for {model_name}")

    # Locate PCA file inside package - only works if the package is installed
    path = files("surgical_embeddings").joinpath("pca_models", PCA_FILES[model_name])

    # path = (
    #     Path(__file__).parent
    #     / "pca_models"
    #     / PCA_FILES[model_name]
    # )

    if not path.exists():
        raise FileNotFoundError(
            f"PCA file not found: {path}"
        )

    return np.load(path)

# Calculate number of components for desired variance
def get_num_components(explained_variance_ratio, variance_percent):
    """
    Determine k principal components needed for desired variance.

    Args:
        explained_variance_ratio (np.ndarray): Explained variance ratio from PCA.
        variance_percent (float): Desired percentage of variance to retain.
    Returns:
        int: Number of principal components needed.
    """

    if not (0 < variance_percent <= 100):
        raise ValueError("variance_percent must be between 0 and 100.")

    cumulative_variance = np.cumsum(explained_variance_ratio)
    target = (variance_percent / 100)

    return np.searchsorted(cumulative_variance,target) + 1