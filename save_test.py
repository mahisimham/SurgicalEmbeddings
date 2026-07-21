from src.surgical_embeddings import (
    generate_embeddings,
    save_embeddings,
)

text = [
    "laparoscopic",
    "robotic",
]

# minilm_embeddings = generate_embeddings(['laparoscopic', 'robotic'], model_name="MiniLM")
# save_embeddings(minilm_embeddings, output_dir="embeddings/minilm")

minilm_embeddings = generate_embeddings(text, model_name="all", apply_pca=True, variance_percent=95)
save_embeddings(minilm_embeddings, output_dir="embeddings")