
#--- 3 strategies---#
#--- 1. TF-IDF: default 
#    2. Key BERT: slower, buat using MMR allows diversity
#    3. Centroid: always produces natural language label ---#

import logging
import re
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class ClusterLabeller:
    

    def __init__(self, config: dict, embedder: Any) -> None:
        self.method: str = config.get("method", "tfidf").lower()
        self.top_n: int = config.get("top_n_keywords", 8)
        self.use_ngrams: bool = config.get("use_ngrams", True)
        self.ngram_range: tuple = tuple(config.get("ngram_range", [1, 2]))
        self.keybert_diversity: float = config.get("keybert_diversity", 0.5)
        self.embedder = embedder


    def label(self, documents: List[str], labels: np.ndarray, embeddings: np.ndarray) -> Dict[int, dict]:
        
        unique_clusters = sorted(set(labels))

        if self.method == "tfidf":
            return self._label_tfidf(documents, labels, embeddings, unique_clusters)

        elif self.method == "keybert":
            return self._label_keybert(documents, labels, embeddings, unique_clusters)

        elif self.method == "centroid":
            return self._label_centroid(documents, labels, embeddings, unique_clusters)

        else:
            logger.warning("Unknown labelling method '%s', falling back to TF-IDF", self.method)
            return self._label_tfidf(documents, labels, embeddings, unique_clusters)


    def _label_tfidf(self, documents: List[str], labels: np.ndarray, embeddings: np.ndarray, unique_clusters: List[int]) -> Dict[int, dict]:
        
        ngram_range = self.ngram_range if self.use_ngrams else (1, 1)

        vectorizer = TfidfVectorizer(
            ngram_range = ngram_range,
            stop_words = "english",
            min_df = 2,           
            max_df = 0.85,       
            sublinear_tf = True,  
        )
        
        vectorizer.fit(documents)
        feature_names = np.array(vectorizer.get_feature_names_out())

        cluster_info = {}
        for cid in unique_clusters:
            cluster_docs_idx = np.where(labels == cid)[0]
            cluster_docs = [documents[i] for i in cluster_docs_idx]

            pseudo_doc = " ".join(cluster_docs)
            tfidf_vec = vectorizer.transform([pseudo_doc]).toarray()[0]

            top_indices = tfidf_vec.argsort()[::-1][: self.top_n]
            keywords = [
                feature_names[i]
                for i in top_indices
                if tfidf_vec[i] > 0
            ]

            
            label_str = self._keywords_to_label(keywords, cid)

    
            cluster_embeddings = embeddings[cluster_docs_idx]
            centroid = cluster_embeddings.mean(axis = 0)

            
            rep_docs = self._representative_docs(cluster_docs, cluster_embeddings, centroid)

            cluster_info[cid] = {
                "label": label_str,
                "keywords": keywords,
                "size": len(cluster_docs),
                "docs": rep_docs,
                "centroid": centroid,
            }

        return cluster_info



    def _label_keybert(self, documents: List[str], labels: np.ndarray,embeddings: np.ndarray,unique_clusters: List[int]) -> Dict[int, dict]:
        
        cluster_info = {}

        for cid in unique_clusters:
            cluster_docs_idx = np.where(labels == cid)[0]
            cluster_docs = [documents[i] for i in cluster_docs_idx]
            cluster_embeddings = embeddings[cluster_docs_idx]
            centroid = cluster_embeddings.mean(axis=0)

            
            full_text = " ".join(cluster_docs[:50])  #---only 50; higher taiking too long---#
            candidates = self._extract_candidates(full_text)

            if not candidates:
                
                logger.debug("No KeyBERT candidates for cluster %d, using TF-IDF", cid)
                keywords = self._tfidf_fallback(cluster_docs, cid)
            else:
                
                candidate_embeddings = self.embedder.encode(candidates)

                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                cand_norms = candidate_embeddings / (
                    np.linalg.norm(candidate_embeddings, axis = 1, keepdims=True) + 1e-8
                )
                similarities = cand_norms @ centroid_norm

                #
                keywords = self._mmr(
                    candidates,
                    similarities,
                    candidate_embeddings,
                    top_n = self.top_n,
                    diversity = self.keybert_diversity,
                )

            label_str = self._keywords_to_label(keywords, cid)
            rep_docs = self._representative_docs(cluster_docs, cluster_embeddings, centroid)

            cluster_info[cid] = {
                "label": label_str,
                "keywords": keywords,
                "size": len(cluster_docs),
                "docs": rep_docs,
                "centroid": centroid,
            }

        return cluster_info


    def _label_centroid(self, documents: List[str], labels: np.ndarray, embeddings: np.ndarray, unique_clusters: List[int]) -> Dict[int, dict]:
        
        cluster_info = {}

        for cid in unique_clusters:
            cluster_docs_idx = np.where(labels == cid)[0]
            cluster_docs = [documents[i] for i in cluster_docs_idx]
            cluster_embeddings = embeddings[cluster_docs_idx]
            centroid = cluster_embeddings.mean(axis = 0)

            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            doc_norms = cluster_embeddings / (
                np.linalg.norm(cluster_embeddings, axis = 1, keepdims = True) + 1e-8
            )
            sims = doc_norms @ centroid_norm
            closest_idx = int(np.argmax(sims))
            closest_doc = cluster_docs[closest_idx]

            first_sentence = closest_doc.split(".")[0][:80].strip()
            label_str = first_sentence if first_sentence else f"Cluster {cid}"

            
            keywords = self._tfidf_fallback(cluster_docs, cid)
            rep_docs = self._representative_docs(cluster_docs, cluster_embeddings, centroid)

            cluster_info[cid] = {
                "label": label_str,
                "keywords": keywords,
                "size": len(cluster_docs),
                "docs": rep_docs,
                "centroid": centroid,
            }

        return cluster_info



    @staticmethod
    def _keywords_to_label(keywords: List[str], cid: int) -> str:
        
        if not keywords:
            return f"Cluster {cid}"
        top = keywords[:3]
        
        return " / ".join(w.title() for w in top)

    @staticmethod
    def _representative_docs(docs: List[str], embeddings: np.ndarray, centroid: np.ndarray, n: int = 5) -> List[str]:
       
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
       
        doc_norms = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )
        sims = doc_norms @ centroid_norm
        top_idx = np.argsort(sims)[::-1][:n]
        return [docs[i] for i in top_idx]

    @staticmethod
    def _extract_candidates(text: str, ngram_range: tuple = (1, 2)) -> List[str]:
        
        STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "to", "of", "and", "in", "for", "on", "at", "by", "with",
            "it", "its", "that", "this", "from", "as", "or", "but",
            "not", "have", "has", "had", "do", "does", "did", "will",
            "can", "could", "would", "should", "may", "might", "shall",
        }
        
        tokens = [
            t.lower()
            for t in re.findall(r"\b[a-zA-Z]{3,}\b", text)
            if t.lower() not in STOPWORDS
        ]
        
        candidates = set()
        
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                candidates.add(phrase)
        
        return list(candidates)[:300]  

    @staticmethod
    def _mmr(candidates: List[str], scores: np.ndarray, candidate_embeddings: np.ndarray, top_n: int,diversity: float) -> List[str]:
        
        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        for _ in range(min(top_n, len(candidates))):
            if not remaining_indices:
                break

            if not selected_indices:
                
                best = max(remaining_indices, key = lambda i: scores[i])
            else:
                
                selected_embs = candidate_embeddings[selected_indices]
                best = max(
                    remaining_indices,
                    key=lambda i: (
                        (1 - diversity) * scores[i]
                        - diversity
                        * float(
                            np.max(
                                candidate_embeddings[i]
                                @ selected_embs.T
                                / (
                                    np.linalg.norm(candidate_embeddings[i]) + 1e-8
                                )
                                / (
                                    np.linalg.norm(selected_embs, axis = 1) + 1e-8
                                )
                            )
                        )
                    ),
                )

            selected_indices.append(best)
            remaining_indices.remove(best)

        return [candidates[i] for i in selected_indices]

    def _tfidf_fallback(self, docs: List[str], cid: int) -> List[str]:
        
        try:
            vec = TfidfVectorizer(
                stop_words = "english",
                ngram_range = self.ngram_range if self.use_ngrams else (1, 1),
                max_features = 500,
            )
            vec.fit(docs)
            pseudo = " ".join(docs)
            tfidf = vec.transform([pseudo]).toarray()[0]
            names = np.array(vec.get_feature_names_out())
            top = tfidf.argsort()[::-1][: self.top_n]
        
            return [names[i] for i in top if tfidf[i] > 0]
        
        except Exception:
            return [f"cluster_{cid}"]
    