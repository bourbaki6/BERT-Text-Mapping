#--- Reduction before clustering is imp. cause 
#    kmeans and HDBSCAN suffer in H D spaces:
#    1. in these spaces distances become numeriucally insignificant
#    ie Curse if dimentionality henec makes it hard for the algo
#    to differnetiate between densse and sparse regions
# 
#    2. UMAP better as it preserves local and global structures
#    better because of non linearilty---#

#--- No 2D because info losss too high for good clusters---#

#--- n = 10; by default---#
  

import logging
from typing import Optional

import numpy as np
import umap

logger = logging.getLogger(__name__)


class DimReducer:

    def __init__(self, config: dict) -> None:
        self.config = config
        self.enabled: bool = config.get("enabled", True)
        self.n_neighbors: int = config.get("n_neighbors", 15)
        self.n_components: int = config.get("n_components", 10)
        self.min_dist: float = config.get("min_dist", 0.0)
        self.metric: str = config.get("metric", "cosine")
        self.random_state: int = config.get("random_state", 42)

        self.viz_n_neighbors: int = config.get("viz_n_neighbors", 15)
        self.viz_n_components: int = config.get("viz_n_components", 2)
        self.viz_min_dist: float = config.get("viz_min_dist", 0.1)

        self._umap_cluster: Optional[umap.UMAP] = None
        self._umap_viz: Optional[umap.UMAP] = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        
        logger.info(
            "Fitting UMAP for clustering: %d -> %d dims "
            "(n_neighbors = %d, min_dist=%.2f, metric=%s)",
            embeddings.shape[1],
            self.n_components,
            self.n_neighbors,
            self.min_dist,
            self.metric,
        )
        self._umap_cluster = umap.UMAP(
            n_neighbors = self.n_neighbors,
            n_components = self.n_components,
            min_dist = self.min_dist,
            metric = self.metric,
            random_state = self.random_state,
            low_memory = False,
            verbose = False,
        )
        reduced = self._umap_cluster.fit_transform(embeddings)
        logger.info("Clustering projection shape: %s", reduced.shape)
        return reduced.astype(np.float32)

    def transform_2d(self, embeddings: np.ndarray) -> np.ndarray:
        
        logger.info(
            "Fitting 2D UMAP for visualisation (n_neighbors=%d, min_dist=%.2f)",
            self.viz_n_neighbors,
            self.viz_min_dist,
        )
        self._umap_viz = umap.UMAP(
            n_neighbors = self.viz_n_neighbors,
            n_components = self.viz_n_components,
            min_dist = self.viz_min_dist,
            metric = self.metric,
            random_state = self.random_state,
            verbose = False,
        )
        viz_reduced = self._umap_viz.fit_transform(embeddings)
        logger.info("Visualisation projection shape: %s", viz_reduced.shape)
        return viz_reduced.astype(np.float32)