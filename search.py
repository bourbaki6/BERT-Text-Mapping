
import argparse
import sys

import numpy as np
import yaml

from embeddings.embed import Embedding
from data.dataset import DataLoader
from semantic_search import SemanticSearch
from logging_utils import setup_logging
import logging
import json
from pathlib import Path
 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description = "Semantic search over clustered documents")
    p.add_argument("--query", required = True, help = "Search query text")
    p.add_argument("--top-k", type = int, default = 10, help = "Number of results to return")
    p.add_argument("--cluster", type = int, default = None, help = "Restrict search to this cluster ID")
    p.add_argument("--config", default = "config.yaml", help ="Path to config file")
    return p.parse_args()


def main() -> None:
    setup_logging(logging.WARNING)  
    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    cache_dir = Path(config["embeddings"]["cache_path"]).parent
    cache_files = list(cache_dir.glob("embeddings_*.npy"))
    if not cache_files:
        print("[ERROR] No cached embeddings found. Run main.py first.")
        sys.exit(1)

    embeddings = np.load(cache_files[0])
    print(f"Loaded embeddings: {embeddings.shape}")

    loader = DataLoader(config["data"])
    documents = loader.load()

    results_dir = Path(config["output"]["results_dir"])
    detail_dir = results_dir / "cluster_details"
    
    if not detail_dir.exists():
        print("[ERROR] No cluster results found. Run main.py first.")
        sys.exit(1)

    cluster_info = {}
    labels_list = [-1] * len(documents)
    for json_file in sorted(detail_dir.glob("cluster_*.json")):
        with open(json_file) as fh:
            info = json.load(fh)
        cid = int(json_file.stem.replace("cluster_", "").replace("noise", "-1"))
        cluster_info[cid] = info

    doc_map_path = results_dir / "document_map.csv"
    if doc_map_path.exists():
        import csv
        with open(doc_map_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                labels_list[int(row["doc_id"])] = int(row["cluster_id"])
    labels = np.array(labels_list)


    embedder = Embedding(config["embeddings"])
    ss = SemanticSearch(embedder, documents, embeddings, labels, cluster_info)

    print(f'\nSearching for: "{args.query}"')
    if args.cluster is not None:
        print(f"Restricted to cluster: {args.cluster}")
    print("─" * 60)

    results = ss.search(args.query, top_k=args.top_k, cluster_id=args.cluster)
    ss.print_results(results)


if __name__ == "__main__":
    main()