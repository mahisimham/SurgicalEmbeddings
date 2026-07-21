from pathlib import Path
import numpy as np

def save_embeddings(embedding_results, output_dir):
    """
    Save embeddings as individual npz files.

    Naming:
        Model_embeddings_original.npz
        Model_embeddings_pca_full.npz
        Model_embeddings_pca_95percent.npz

    Args:
        embedding_results (dict): Dictionary containing embeddings and metadata.
        output_dir (str or Path): Directory to save the embedding files. 
    Returns:
        None
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = embedding_results["embeddings"]
    metadata = embedding_results["metadata"]

    pca_applied = metadata["pca_applied"]
    variance_percent = metadata["variance_percent"]

    if not pca_applied:
        suffix = "original"
    elif variance_percent is None:
        suffix = "pca_full"
    else:
        suffix = (f"pca_{variance_percent}percent")

    for model, array in embeddings.items():
        filename = (f"{model}_embeddings_{suffix}.npz")
        filepath = (output_dir / filename)

        np.savez(filepath, embeddings=array)

        print(f"Saved: {filepath}")

# import os
# import numpy as np

# def save_embeddings(embeddings, output_dir):
#     """
#     Save embeddings separately for each model.

#     Args:
#         embeddings: dictionary of model names and their corresponding embeddings.
#         output_dir: directory to save the embedding files.
#     Returns:
#         None
#     """

#     os.makedirs(output_dir, exist_ok=True)

#     for model_name, embedding in embeddings.items():
#         file_path = os.path.join(output_dir, f"{model_name}_embeddings.npz")

#         np.savez_compressed(
#             file_path,
#             embeddings=embedding,
#             model=model_name,
#         )