
# VibeMatcher 360 Vision 

## Overview

**VibeMatcher 360 Vision** is a semantic fashion recommendation prototype that translates natural-language “vibe” queries into meaningful product matches using vector embeddings.
It integrates **OpenAI text embeddings** with **visual analytics (UMAP + heatmaps)** and **auto-tag generation** through clustering, providing both interpretability and intelligent catalog enrichment.

This project demonstrates how AI can enhance product discovery by mapping aesthetic intent into data-driven, explainable recommendations.

---

## Objectives

* Generate text embeddings for product descriptions using OpenAI’s `text-embedding-ada-002` model.
* Enable natural-language vibe search using cosine similarity.
* Visualize embedding relationships through UMAP projections and similarity heatmaps.
* Enrich catalog metadata automatically by clustering semantic vectors to suggest new vibe tags.
* Evaluate similarity quality and query latency to assess performance.

---

## Features

### 1. Data Preparation

* Dataset of 8 curated fashion products with descriptions and base vibe tags.
* Structured using `pandas` for extensibility and reproducibility.

### 2. Embedding Generation

* Uses **OpenAI’s text-embedding-ada-002** for vectorization.
* Includes secure API key input (`getpass()`), caching, and fallback to simulated embeddings.
* Embeddings stored locally for reusability.

### 3. Vector Search (Cosine Similarity)

* Computes semantic similarity between query and product embeddings using `sklearn.metrics.pairwise.cosine_similarity`.
* Retrieves **top-3 ranked matches** with similarity scores.
* Includes threshold-based fallback for low-confidence matches.

### 4. Evaluation and Metrics

* Tested across multiple queries (*energetic urban chic*, *cozy boho weekend*, *minimal workwear sleek*).
* Captures latency, similarity score, and quality classification.
* Produces modern latency visualization using Seaborn.

---

## Innovation Highlights

| Innovation                             | Description                                                                                                                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **UMAP + Heatmap Visualization**       | Projects high-dimensional product embeddings into a 2D “vibe space” using UMAP and plots inter-product similarity via heatmap. Provides visual interpretability of catalog structure. |
| **Auto-Tag Generation via Clustering** | Uses K-Means clustering on embeddings to identify latent style groups and automatically suggest new vibe tags for unlabelled or weakly tagged products.                               |

These innovations combine **explainable AI** with **unsupervised enrichment**, creating a more intelligent and visually interpretable recommendation engine.

---

## System Flow

```
Product Descriptions  →  Embeddings (OpenAI / Simulated)
                                   ↓
                           UMAP Visualization
                                   ↓
                        Cosine Similarity Search
                                   ↓
                        Top-3 Product Recommendations
                                   ↓
                  Clustering → Auto-Tag Generation
```

---

## Technical Stack

| Component          | Technology                                                   |
| ------------------ | ------------------------------------------------------------ |
| **Language**       | Python 3.10+                                                 |
| **Environment**    | Google Colab / Jupyter Notebook                              |
| **Core Libraries** | pandas, numpy, scikit-learn, matplotlib, seaborn, umap-learn |
| **Model**          | text-embedding-ada-002 (OpenAI)                              |
| **Visualization**  | Seaborn + Matplotlib                                         |
| **Clustering**     | KMeans                                                       |

---

## Evaluation Summary

| Query                  | Top Match           | Similarity | Latency (s) | Quality |
| ---------------------- | ------------------- | ---------- | ----------- | ------- |
| energetic urban chic   | Urban Bomber Jacket | 0.86       | 2.7         | Good    |
| cozy boho weekend      | Boho Dress          | 0.78       | 2.1         | Good    |
| minimal workwear sleek | Chic Midi Skirt     | 0.71       | 1.6         | Good    |

Average Similarity ≈ 0.78  Average Latency ≈ 2.1 s

---

## Reflection

* Demonstrated semantic retrieval using text embeddings.
* Visualized relationships between fashion products in embedding space.
* Achieved interpretable clustering for automatic vibe tag enrichment.
* Maintained robustness through hybrid embedding fallback.
* Established a foundation for scalable vector-based recommendation systems.

---

## Future Work

* Integrate **Pinecone / FAISS** for large-scale vector indexing.
* Extend to **multimodal CLIP embeddings** (image + text).
* Build an interactive **Streamlit dashboard** for real-time vibe exploration.
* Incorporate **user feedback loops** for personalized re-ranking.

---

## Repository Structure

```
📂 vibematcher-vision/
 ├── vibe_matcher_ram.ipynb          # Main notebook
 ├── README.md                       # Documentation (this file)
 ├── requirements.txt                 # Dependency list
 ├── product_embeddings_cache.json    # Cached embeddings (optional)
 └── /outputs                         # Visualizations & evaluation results
```

---

## License

Distributed under the **MIT License**.
Use of OpenAI models complies with the [OpenAI API Terms of Service](https://openai.com/policies/terms-of-use).

---

## Author

**Ramkumar R**
AI and Data Science Research Enthusiast

