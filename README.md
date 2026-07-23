# SurgicalEmbeddings

Generate text embeddings for surgical and medical terms using pretrained language models, with optional dimensionality reduction via precomputed PCA transforms.

## Overview

`surgical_embeddings` wraps three HuggingFace models behind a single interface:

| Key | Model |
|---|---|
| `SapBERT` | [`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) |
| `MiniLM` | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| `BGE_Large` | [`BAAI/bge-large-zh-v1.5`](https://huggingface.co/BAAI/bge-large-zh-v1.5) |

Given a list of strings, it produces raw embeddings for one, several, or all of these models, and can optionally project them through a precomputed PCA transform (full-rank, or truncated to retain a target percentage of variance).

## Project structure

```
src/surgical_embeddings/
├── embed.py          # generate_embeddings() and per-model embedding logic
├── io.py             # save_embeddings() — writes .npz files per model/config
├── pca.py            # apply_full_pca() — applies precomputed PCA matrices
└── pca_models/        # precomputed PCA components (mean, components, explained variance) per model
tests/
└── test_embeddings.py # pytest suite covering model selection, PCA, saving, and error handling
example_usage.py       # end-to-end usage examples
requirements.txt       # pinned dependencies
```

`embeddings/` is the default output directory for generated `.npz` files; it's git-ignored since embeddings are generated artifacts, not source.

## Environment

- Python 3.12 (developed/tested against 3.12.7)
- Dependencies pinned in [`requirements.txt`](requirements.txt) — key ones: `torch`, `transformers`, `sentence-transformers`, `numpy`, `scikit-learn`

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The package isn't installed/packaged (no build backend configured yet), so run scripts and tests from the repository root so `src/` resolves on the path — see the examples below.

## Usage

### Generate embeddings for all models

```python
from src.surgical_embeddings import generate_embeddings, save_embeddings

text = ["laparoscopic", "robotic"]

result = generate_embeddings(text, model_name="all")
save_embeddings(result, output_dir="embeddings/all")
```

`result["embeddings"]` is a dict of `{model_name: np.ndarray}`; `result["metadata"]` records whether PCA was applied.

### Generate embeddings with PCA, retaining 95% of variance

```python
from src.surgical_embeddings import generate_embeddings, save_embeddings

result = generate_embeddings(
    ["laparoscopic", "robotic"],
    model_name="MiniLM",       # or a list, e.g. ["MiniLM", "SapBERT"]
    apply_pca=True,
    variance_percent=95,        # omit for full-rank PCA
)
save_embeddings(result, output_dir="embeddings/minilm")
```

See [`example_usage.py`](example_usage.py) for more.

## Testing

```bash
pytest
```

## License

See [`LICENSE`](LICENSE).
