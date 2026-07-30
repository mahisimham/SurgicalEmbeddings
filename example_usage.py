from surgical_embeddings import (
    generate_embeddings,
    save_embeddings,
    plot_pca
)

"""
Parameter info:
- text: List of strings to generate embeddings for
ex: ["laparoscopic", "robotic"]
- model_name: Model name(s) from HuggingFace to use for generating embeddings. 
It can be a single model, a list of models, or "all" to use all available models
ex: "all", "MiniLM", "SapBERT", "BGE_Large"
- (optional) apply_pca: boolean flag to indicate whether to apply PCA transformation to the embeddings
ex: True, False
- (optional but required if apply_pca is True) variance_percent: The percentage of variance to retain if PCA is applied.
ex: 95, 99, None (for full PCA)
"""


text = [
    "laparoscopic",
    "robotic",
]

# Generate and save embeddings for all models, no PCA or variance applied
all_embeddings = generate_embeddings(text, model_name="all")
save_embeddings(all_embeddings, output_dir="embeddings/all")

# Generate and save embeddings for all models, full PCA applied
all_embeddings = generate_embeddings(text, model_name="all", apply_pca=True)
save_embeddings(all_embeddings, output_dir="embeddings/all")

# Generate and save embeddings for all models, PCA applied, 95% variance retained
all_embeddings = generate_embeddings(text, model_name="all", apply_pca=True, variance_percent=95)
save_embeddings(all_embeddings, output_dir="embeddings/all")

# Plot PCA
plot_pca("embeddings/all/BGE_Large_embeddings_pca_full.npz", title="BGE Large PCA Embeddings")
