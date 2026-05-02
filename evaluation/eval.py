
from typing import Dict

import numpy as np
from sklearn import metrics

import logging
logger = logging.getLogger(__name__)


class Evaluator:
    

    def __init__(self, config: dict) -> None:
        self.compute_silhouette: bool = config.get("silhouette", True)
        self.compute_davies_bouldin: bool = config.get("davies_bouldin", True)
        self.compute_calinski: bool = config.get("calinski_harabasz", True)
        self.max_sample: int = config.get("max_sample_for_metrics", 5000) or 999_999


    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        
        valid_mask = labels != -1
        valid_embs = embeddings[valid_mask]
        valid_labels = labels[valid_mask]

        n_valid = len(valid_labels)
        n_clusters = len(set(valid_labels))

        if n_clusters < 2:
            logger.warning(
                "Only %d valid cluster(s) — cannot compute metrics (need ≥ 2).", n_clusters
            )
            return {"error": "insufficient_clusters"}

    
        if n_valid > self.max_sample:
            logger.info(
                "Subsampling %d -> %d documents for metric computation",
                n_valid,
                self.max_sample,
            )
            rng = np.random.default_rng(42)
            idx = rng.choice(n_valid, size=self.max_sample, replace=False)
            valid_embs = valid_embs[idx]
            valid_labels = valid_labels[idx]

        results: Dict[str, float] = {
            "n_documents": float(n_valid),
            "n_clusters": float(n_clusters),
            "noise_points": float(int(np.sum(labels == -1))),
        }

        if self.compute_silhouette:
            try:
                score = metrics.silhouette_score(
                    valid_embs,
                    valid_labels,
                    metric = "euclidean",
                    sample_size = min(n_valid, 3000),  
                    random_state = 42,
                )
                results["silhouette_score"] = float(score)
                logger.debug("Silhouette score: %.4f", score)
            
            except Exception as exc:
                logger.warning("Could not compute silhouette score: %s", exc)

    
        if self.compute_davies_bouldin:
            try:
                score = metrics.davies_bouldin_score(valid_embs, valid_labels)
                results["davies_bouldin_index"] = float(score)
                logger.debug("Davies-Bouldin index: %.4f", score)
            except Exception as exc:
                logger.warning("Could not compute Davies-Bouldin index: %s", exc)

        
        if self.compute_calinski:
            
            try:
                score = metrics.calinski_harabasz_score(valid_embs, valid_labels)
                results["calinski_harabasz_score"] = float(score)
                logger.debug("Calinski-Harabasz score: %.4f", score)
            
            except Exception as exc:
                logger.warning("Could not compute Calinski-Harabasz score: %s", exc)

    
        cluster_sizes = [
            int(np.sum(valid_labels == c)) for c in set(valid_labels)
        ]
        
        results["avg_cluster_size"] = float(np.mean(cluster_sizes))
        results["min_cluster_size"] = float(np.min(cluster_sizes))
        results["max_cluster_size"] = float(np.max(cluster_sizes))
        results["cluster_size_std"] = float(np.std(cluster_sizes))

        return results

    def interpret(self, metrics_dict: Dict[str, float]) -> str:
        
        lines = []
        sil = metrics_dict.get("silhouette_score")
        db = metrics_dict.get("davies_bouldin_index")
        ch = metrics_dict.get("calinski_harabasz_score")

        if sil is not None:
            
            if sil > 0.5:
                quality = "excellent"
            
            elif sil > 0.3:
                quality = "good"
            
            elif sil > 0.1:
                quality = "moderate"
            
            else:
                quality = "poor — consider tuning min_cluster_size or UMAP parameters"
            lines.append(f"  Silhouette ({sil:.4f}): {quality}")

        if db is not None:
            
            if db < 0.5:
                quality = "excellent"
            elif db < 1.0:
                quality = "good"
            else:
                quality = "moderate — may overlap"
            lines.append(f"Davies-Bouldin ({db:.4f}): {quality}")

        if ch is not None:
            lines.append(f"Calinski-Harabasz ({ch:.1f}): higher is better")

        return "\n".join(lines) if lines else "  No metrics available."