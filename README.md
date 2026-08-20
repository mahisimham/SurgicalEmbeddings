# SurgicalEmbeddings

Generate normalized text embeddings for surgical and medical terms using pretrained language models, with optional dimensionality reduction via precomputed PCA transforms and plotting of saved PCA embeddings.

## Overview

`surgical_embeddings` wraps three HuggingFace models behind a single interface and provides embeddings for surgical procedures:

| Key | Model |
|---|---|
| `SapBERT` | [`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) |
| `MiniLM` | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| `BGE_Large` | [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) |

Given a list of strings, a single string, or a path to a `.txt` file (one term per line), it produces normalized embeddings for one, several, or all of these models, and can optionally project them through a precomputed PCA transform (full-rank, or truncated to retain a target percentage of variance).

## Installation

SurgicalEmbeddings requires Python 3.10 or later. It can be installed from PyPI with:

```bash
pip install surgical-embeddings
```

Or directly from GitHub and create a virtual environment:

```bash
python -m pip install "git+https://github.com/mahisimham/SurgicalEmbeddings.git"

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
```

The pretrained Hugging Face models are downloaded and cached automatically
when each model is used for the first time.

## Project structure

```
src/surgical_embeddings/
├── embed.py          # generate_embeddings(), load_text(), and per-model embedding logic
├── io.py             # save_embeddings() — writes .npz files per model/config
├── pca.py            # apply_full_pca() — applies precomputed PCA matrices
├── plot.py           # plot_pca() — saves a PNG visualization of PCA embeddings
└── pca_models/        # precomputed PCA components (mean, components, explained variance) per model
tests/
└── test_embeddings.py # pytest suite covering model selection, PCA, saving, and error handling
usage/
├── example_usage.py   # end-to-end usage examples, using the installed package
└── test_input.txt     # sample input file (one term per line)
requirements.txt       # pinned dependencies
```

`embeddings/` is the default output directory for generated `.npz` files

## Environment

- Python 3.10 or later 
- Runtime dependencies are declared in [`pyproject.toml`](pyproject.toml)
- A fully pinned development environment is available in [`requirements.txt`](requirements.txt)

For the fully pinned development environment, use
`python -m pip install -r requirements.txt`.

## Usage

### Generate and save embeddings

```python
from surgical_embeddings import (
    generate_embeddings,
    save_embeddings,
)

# Option 1: a list of terms
terms = ["laparoscopic", "robotic"]
embeddings = generate_embeddings(terms, model_name="all")
save_embeddings(embeddings, output_dir="embeddings/all")

# Option 2: a path to a .txt file with one term per line
embeddings = generate_embeddings("test_input.txt", model_name="all")
save_embeddings(embeddings, output_dir="embeddings/all")
```

`model_name` accepts a single model key, a list such as
`["MiniLM", "SapBERT"]`, or `"all"`. In place of the list of terms, a single
string is also accepted, as is a path to a `.txt` file with one term per
line (blank lines are skipped).

The returned value contains:

- `embeddings`: a dictionary mapping each selected model key to a two-dimensional NumPy array
- `metadata`: the `pca_applied` and `variance_percent` settings used for the run

The example above creates one file per model:
`<Model>_embeddings_original.npz`. Each archive stores its array under the
`embeddings` key.

### Apply PCA

```python
from surgical_embeddings import generate_embeddings, save_embeddings

result = generate_embeddings(
    ["laparoscopic", "robotic"],
    model_name="MiniLM",
    apply_pca=True,
    variance_percent=95,
)
save_embeddings(result, output_dir="embeddings/minilm")
```

Set `apply_pca=True` without `variance_percent` to use the full precomputed PCA
projection. Set `variance_percent` to a value greater than 0 and at most 100 to
retain the number of principal components needed to reach that percentage of
explained variance.

PCA projection loads its precomputed matrices via `importlib.resources`, so
`apply_pca=True` requires `surgical_embeddings` to actually be installed
(`pip install .` or `pip install surgical-embeddings`) rather than just added
to `sys.path`.

Saved files use the configuration in their names:

| Configuration | Filename suffix |
|---|---|
| No PCA | `_embeddings_original.npz` |
| Full PCA | `_embeddings_pca_full.npz` |
| 95% retained variance | `_embeddings_pca_95percent.npz` |

### Plot saved PCA embeddings

```python
from surgical_embeddings import plot_pca

plot_pca(
    "embeddings/all/BGE_Large_embeddings_pca_full.npz",
    title="BGE Large PCA Embeddings",
)
```

`plot_pca` reads the `embeddings` array from the `.npz` archive and saves the
plot alongside it as
`BGE_Large_embeddings_pca_full_plot.png`. The saved embeddings must have at
least two dimensions. Pass `title=None` to use the default title, `PCA
Embeddings`.

See [`usage/example_usage.py`](usage/example_usage.py) for an end-to-end example that
generates original, full-PCA, and 95%-variance embeddings before plotting the
full-PCA BGE output.

## Testing

```bash
pytest
```

## License

See [`LICENSE`](LICENSE).
