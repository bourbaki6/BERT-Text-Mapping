#--- more meaning than exat keyword match---#

import logging
from typing import Dict, List, Optional 

import numpy as np
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class SemanticSearch:
    

    def __init__(self,embedder,documents: List[str], embeddings: np.ndarray, labels: np.ndarray, cluster_info: Dict[int, dict]) -> None:
        self.embedder = embedder
        self.documents = documents
        self.embeddings = normalize(embeddings, norm = "l2") 
        self.labels = labels
        self.cluster_info = cluster_info


    def search(self, query: str, top_k: int = 10, cluster_id: Optional[int] = None) -> List[dict]:
       
        query_vec = self.embedder.encode_single(query)      
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)  

        if cluster_id is not None:
            mask = self.labels == cluster_id
            
            if not np.any(mask):
                logger.warning("No documents found in cluster %d", cluster_id)
                return []
            search_embeddings = self.embeddings[mask]
            search_indices = np.where(mask)[0]
        else:
            search_embeddings = self.embeddings
            search_indices = np.arange(len(self.documents))

        scores = search_embeddings @ query_vec 

        top_local_indices = np.argsort(scores)[::-1][:top_k]
        top_global_indices = search_indices[top_local_indices]

        results = []
        for rank, (local_idx, global_idx) in enumerate(
            zip(top_local_indices, top_global_indices), start = 1
        ):
            cid = int(self.labels[global_idx])
            cluster_label = self.cluster_info.get(cid, {}).get("label", "Unknown")
            results.append({
                "rank": rank,
                "score": float(scores[local_idx]),
                "doc_id": int(global_idx),
                "cluster_id": cid,
                "cluster_label": cluster_label,
                "document": self.documents[global_idx],
            })

        return results

    def search_by_cluster(self, query: str, top_k_per_cluster: int = 3) -> Dict[int, List[dict]]:
        
        all_cluster_ids = sorted(set(self.labels))
        output = {}
        for cid in all_cluster_ids:
            results = self.search(query, top_k = top_k_per_cluster, cluster_id = cid)
            if results:
                output[cid] = results
        return output

    def print_results(self, results: List[dict]) -> None:
        
        if not results:
            print("No results found.")
            return
        for r in results:
            print(
                f"[{r['rank']:2d}] Score: {r['score']:.3f}  "
                f"| Cluster {r['cluster_id']}: {r['cluster_label']}"
            )
            print(f"{r['document'][:150]}{'…' if len(r['document']) > 150 else ''}")
            print()