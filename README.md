# BERT Text Mapping

Automatic topic discovery for text corpora without any labels. The project embeds every document with Sentence-BERT, compresses into a lower-dimensional space with UMAP, and clusters with HDBSCAN.

## Why SBERT and not BERT?
 
Raw BERT pooling produces embeddings that cluster badly — the model wasn't trained with any objective that makes cosine similarity meaningful. SBERT was fine-tuned with a siamese/triplet network specifically so that semantic similarity maps to cosine distance. This makes it a better fit for this task.

## Pipeline
 
Seven stages, each in its own module:
 
```
Load corpus -> SBERT encode -> UMAP reduce -> HDBSCAN cluster -> Label -> Evaluate -> Export
```
 
UMAP runs twice: once to a higher-dimensional space for clustering (default 10 components), and again to 2D purely for the visualisation. The clustering stage uses the 10-D reduction; the scatter plot uses the 2-D one.

 
## Results on the demo corpus
 
300 documents across 10 topic areas. HDBSCAN found all 10 clusters with no noise points.
 
| Metric | Score 
|---|---|
| Silhouette | 0.4072 |
| Davies-Bouldin | 0.7688 |
| Calinski-Harabasz | 80.98 |
| Noise points | 0 | 
| Avg cluster size | 30 docs | 
 
Cluster 0 ended up absorbing 186 of the 300 documents — a known limitation of running on a small, synthetic corpus where several topic areas share surface vocabulary (climate, medicine, and space all use words like "radiation", "energy", "heat"). On a real corpus this flattens out.

## Labelling methods
 
Three options, trade-off between speed and quality:
 
**`tfidf`** (default) — builds a pseudo-document per cluster and extracts the top TF-IDF unigrams and bigrams. Fast, works well on most corpora.
 
**`keybert`** — embeds n-gram candidates, ranks them by cosine similarity to the cluster centroid, then applies MMR (Maximal Marginal Relevance) to keep the keyword set diverse. Slower but produces noticeably better labels, especially when clusters are semantically broad.
 
**`centroid`** — finds the document closest to the centroid and uses its first sentence as the label. Good when you want a natural language summary rather than a keyword list.
