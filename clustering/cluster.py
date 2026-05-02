#---2 Algorithms  Used:
#
#  i> HDSCAN - just as a default - supervised learning
#  ii> K means - as teh alt.:
#      no noise, all documents just get one cluster for every document


import hdbscan
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


import logging

logger = logging.getLogger(__name__)

from typing import Optional

import numpy as np

class Cluster:

    def __init__(self, config: dict) -> None:
        
        self.method: str = config.get("method", "hdbscan").lower()
        self.hdbscan_cfg: dict = config.get("hdbscan", {})
        self.kmeans_cfg: dict = config.get("kmeans", {})

    def fit(self, embeddings: np.ndarray) -> np.ndarray:

        if self.method == "hdbscan":
            return self._run_hdbscan(embeddings)

        elif self.method == "kmeans":
            return self._run_kmeans(embeddings)

        else: 
            raise ValueError(f"Unknown clustering method: '{self.method}'")


    def _run_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:

        cfg = self.hdbscan_cfg
        min_cluster_size: int = cfg.get("min_cluster_size", 10)
        min_samples: Optional[int] = cfg.get("min_samples")  # None → HDBSCAN default
        metric: str = cfg.get("metric", "euclidean")
        noise_handling: str = cfg.get("noise_handling", "nearest")
 
        logger.info(
            "Running HDBSCAN  (min_cluster_size = %d, min_samples = %s, metric=%s)",
            min_cluster_size,
            min_samples,
            metric,
        )
 
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size = min_cluster_size,
            min_samples = min_samples,
            metric =metric,
            cluster_selection_method = "eom",
            prediction_data = True,   
        )
        labels: np.ndarray = clusterer.fit_predict(embeddings)
 
        n_noise = int(np.sum(labels == -1))
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(
            "HDBSCAN result: %d clusters, %d noise points (%.1f%%)",
            n_clusters,
            n_noise,
            100 * n_noise / len(labels),
        )
 
        if n_noise > 0:
            labels = self._handle_noise(labels, embeddings, noise_handling)
            remaining_noise = int(np.sum(labels == -1))
            logger.info(
                "After noise handling (%s): %d noise points remaining",
                noise_handling,
                remaining_noise,
            )
 
        return labels
 
    def _handle_noise(self, labels: np.ndarray, embeddings: np.ndarray, strategy: str) -> np.ndarray:
        
        if strategy == "own_cluster" or strategy == "discard":
            
            return labels
 
        if strategy == "nearest":
            return self._reassign_to_nearest_centroid(labels, embeddings)
 
        logger.warning("Unknown noise_handling strategy '%s' — keeping noise as-is", strategy)
        return labels
 
    @staticmethod
    def _reassign_to_nearest_centroid(
        labels: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        
        labels = labels.copy()
        unique_clusters = [c for c in set(labels) if c != -1]
 
        if not unique_clusters:
            logger.warning("No valid clusters to reassign noise to.")
            return labels
 
        centroids = np.array(
            [embeddings[labels == c].mean(axis = 0) for c in unique_clusters]
        )  

        centroids_norm = normalize(centroids, norm = "l2")
 
        noise_mask = labels == -1
        noise_embeddings = embeddings[noise_mask]       
        noise_norm = normalize(noise_embeddings, norm = "l2")
 
        
        similarities = noise_norm @ centroids_norm.T
        nearest_cluster_idx = np.argmax(similarities, axis=1)
        nearest_cluster_labels = np.array(unique_clusters)[nearest_cluster_idx]
 
        labels[noise_mask] = nearest_cluster_labels
        return labels
 

 
    def _run_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        
        cfg = self.kmeans_cfg
        n_clusters: int = cfg.get("n_clusters", 10)
        n_init: int = cfg.get("n_init", 10)
        max_iter: int = cfg.get("max_iter", 300)
        random_state: int = cfg.get("random_state", 42)
 
        logger.info(
            "Running K-Means  (k = %d, n_init = %d, max_iter = %d)",
            n_clusters,
            n_init,
            max_iter,
        )
 
        km = KMeans(
            n_clusters = n_clusters,
            n_init = n_init,
            max_iter = max_iter,
            random_state = random_state,
            
            init = "k-means++",
        )
        labels: np.ndarray = km.fit_predict(embeddings)
 
        logger.info(
            "K-Means inertia: %.2f  (lower = tighter clusters)", km.inertia_
        )  
        return labels