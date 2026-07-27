import numpy as np
import torch

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from .pca import apply_full_pca

# Constants
MODEL_NAMES = {
    "SapBERT": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "BGE_Large": "BAAI/bge-large-en-v1.5",
}

BATCH_SIZE = 128
MAX_LENGTH = 25

# Main user function to generate embeddings
def generate_embeddings(
    input_list,
    model_name="all",
    apply_pca=False,
    variance_percent=None,
):
    """
    Generate embeddings from selected models.

    Args:
        input_list (list): List of strings to generate embeddings for.
        model_name (str or list): Model name(s) from HuggingFace to use for generating embeddings.
        apply_pca (bool): Whether to apply PCA to the embeddings.
        variance_percent (float): The percentage of variance to retain if PCA is applied.

    Returns:
        dict:
            {
                model_name: np.ndarray
            }
    """

    input_list = validate_input(input_list)
    models = select_models(model_name)
    embeddings = {}

    for model in models:
        print(f"Generating embeddings: {model}")
        model_embeddings = get_embedding(model, input_list)

        if apply_pca:
            if variance_percent is not None:
                if not (0 < variance_percent <= 100):
                    raise ValueError("variance_percent must be between 0 and 100")
            model_embeddings = apply_full_pca(model_embeddings, model, variance_percent)
        
        embeddings[model] = model_embeddings

    metadata = {
        "pca_applied": apply_pca,
        "variance_percent": variance_percent,
    }

    return {
        "embeddings": embeddings,
        "metadata": metadata,
    }

# Input validation 
def validate_input(input_list):
    """
    Validate the input list to ensure it is a string or a list of strings.

    Args:
        input_list (str or list): Input to validate.
    Returns:
        list: Validated list of strings.
    """
    if isinstance(input_list, str):
        return [input_list]

    if isinstance(input_list, list):
        if all(isinstance(x, str) for x in input_list):
            return input_list

    raise ValueError("Input must be a string or list of strings.")

# Select the desired models
def select_models(model_name):
    """
    Select the appropriate models based on the model_name parameter.

    Args:
        model_name (str or list): Model name(s) from HuggingFace to use for generating embeddings.
    Returns:
        list: List of model names to use for generating embeddings.
    """
    if model_name == "all":
        return list(MODEL_NAMES.keys())

    if isinstance(model_name, str):
        if model_name in MODEL_NAMES:
            return [model_name]
        else:
            raise ValueError(f"Unknown model {model_name}")

    if isinstance(model_name, list):
        for model in model_name:
            if model not in MODEL_NAMES:
                raise ValueError(f"Unknown model {model}")

        return model_name

    raise ValueError("Invalid model_name")

# Get embeddings from the corresponding model
def get_embedding(model_name, input_list):
    """
    Route to model-specific embedding function.

    Args:
        model_name (str): Model name from HuggingFace to use for generating embeddings.
        input_list (list): List of strings to generate embeddings for.
    Returns:
        np.ndarray: Generated embeddings.
    """

    if model_name in MODEL_NAMES:
        return sentence_transformer_embedding(model_name, input_list)
    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

# Implementation of SentenceTransformer embedding generation
def sentence_transformer_embedding(model_name, input_list):
    """
    Generate SentenceTransformer embeddings.

    Args:
        model_name (str): Model name from HuggingFace to use for generating embeddings.
        input_list (list): List of strings to generate embeddings for.
    Returns:
        np.ndarray: Generated embeddings.
    """

    model = SentenceTransformer(MODEL_NAMES[model_name])

    if model_name == "BGE_Large":
        embeddings = model.encode(
            input_list,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    elif model_name == "MiniLM":
        embeddings = model.encode(
            input_list,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
        )
    elif model_name == "SapBERT":
        embeddings = model.encode(
            input_list,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return np.asarray(embeddings)