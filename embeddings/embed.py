#--- SBERT: Sentence Embedder - wrapping sentence 
# transformers to produce dense vector embeddings from text----#
#--- No BERT because I'm curious to try something new, 
#    and raw BERT actually pooling produces embeddingsthat cluster badly
#    not worthwhile as SBERT was specifically trained 
#    with a siamese/triplet network where semantic = cosine similarity
#   

import hashlib
import logging
from pathlib import Path
from typing import List, Optional
 
import numpy as np

from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

class Embedding:

    def __init__(self, config: dict) -> None:
        
        self.model_name: str = config["model_name"]
        self.batch_size: int = config.get("batch_size", 64)
        self.normalize: bool = config.get("normalize", True)
        self.cache_embeddings: bool = config.get("cache_embeddings", True)
        self.cache_path: Path = Path(config.get("cache_path", "cache/embeddings.npy"))
        self._model: Optional[SentenceTransformer] = None

    @property 
    def model(self) -> SentenceTransformer:
        
        if self._model is None:
            logger.info("Loading SBERT model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model
 
    def encode(self, documents: List[str]) -> np.ndarray:
        
        cache_key = self._cache_key(documents)
        cache_file = self.cache_path.parent / f"{self.cache_path.stem}_{cache_key[:8]}.npy"
 
        if self.cache_embeddings and cache_file.exists():
            logger.info("Loading cached embeddings from: %s", cache_file)
            embeddings = np.load(cache_file)
            logger.info("Loaded embeddings shape: %s", embeddings.shape)
            return embeddings
 
        logger.info(
            "Encoding %d documents with model '%s' (batch_size=%d) …",
            len(documents),
            self.model_name,
            self.batch_size,
        )
        embeddings = self.model.encode(
            documents,
            batch_size = self.batch_size,
            show_progress_bar = True,   
            convert_to_numpy = True,
            normalize_embeddings = False,  
        )
 
        if self.normalize:
            logger.debug("\n Applying L2 normalization t o  embeddings")
            embeddings = normalize(embeddings, norm = "l2")
 
        embeddings = embeddings.astype(np.float32)
 
    
        if self.cache_embeddings:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, embeddings)
            logger.info("Cached embeddings to: %s", cache_file)
 
        return embeddings
 
    def encode_single(self, text: str) -> np.ndarray:
        
        vec = self.model.encode([text], normalize_embeddings = self.normalize)
        return vec[0].astype(np.float32)

 
    @staticmethod
    def _cache_key(documents: List[str]) -> str:
        
        corpus_fingerprint = "|".join(documents[:100])  
        fingerprint_str = f"{corpus_fingerprint}|len={len(documents)}"
        return hashlib.md5(fingerprint_str.encode()).hexdigest()


