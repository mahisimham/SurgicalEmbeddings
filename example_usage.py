from src.surgical_embeddings import (
    generate_embeddings,
    save_embeddings,
)

text = [
    "laparoscopic",
    "robotic",
]

# model_name can be a single model, a list of models, or "all" to use all available models
# "all", "MiniLM", "SapBERT", "BGE_Large"

# Generate and save embeddings for all models, no PCA or variance applied
all_embeddings = generate_embeddings(['laparoscopic', 'robotic'], model_name="all")
save_embeddings(all_embeddings, output_dir="embeddings/all")

# Generate and save embeddings for all models, full PCA applied
all_embeddings = generate_embeddings(text, model_name="all", apply_pca=True)
save_embeddings(all_embeddings, output_dir="embeddings/all")

# Generate and save embeddings for all models, PCA applied, 95% variance retained
all_embeddings = generate_embeddings(text, model_name="all", apply_pca=True, variance_percent=95)
save_embeddings(all_embeddings, output_dir="embeddings/all")