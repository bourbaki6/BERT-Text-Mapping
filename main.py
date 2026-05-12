
import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

from data.dataset import DataLoader
from embeddings.embed import Embedding
from reductions.umap import DimReducer
from clustering.cluster import Cluster
from labelling.labels import ClusterLabeller
from evaluation.eval import Evaluator
from exporter.exports import Exporter
from visualisation.visuals import Visualiser
from logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    
    p = argparse.ArgumentParser(description="BERT Text Mapping — unsupervised topic discovery")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--input", default=None, help="Override data.input_path in config")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    t_start = time.time()


    banner("Stage 1 / 7 — Loading configuration")
    config = load_config(args.config)
    if args.input:
        config["data"]["input_path"] = args.input
    logger.info("Config loaded from: %s", args.config)

    banner("Stage 2 / 7 — Loading corpus")
    loader = DataLoader(config["data"])
    documents = loader.load()
    if not documents:
        logger.error("No documents loaded — check data.input_path in config.")
        sys.exit(1)
    logger.info("Corpus size: %d documents", len(documents))

    banner("Stage 3 / 7 — Generating SBERT embeddings")
    embedder = Embedding(config["embeddings"])
    embeddings = embedder.encode(documents)
    logger.info("Embeddings shape: %s", embeddings.shape)

    banner("Stage 4 / 7 — UMAP dimensionality reduction")
    reducer = DimReducer(config["reduction"])
    if reducer.enabled:
        reduced = reducer.fit_transform(embeddings)
        viz_embeddings = reducer.transform_2d(embeddings)
    else:
        logger.info("Reduction disabled — clustering on raw embeddings")
        reduced = embeddings
        viz_embeddings = embeddings[:, :2]

    banner("Stage 5 / 7 — Clustering")
    clusterer = Cluster(config["clustering"])
    labels = clusterer.fit(reduced)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info("Clusters found: %d", n_clusters)

    banner("Stage 6 / 7 — Labelling clusters")
    labeller = ClusterLabeller(config["labelling"], embedder)
    cluster_info = labeller.label(documents, labels, embeddings)

    banner("Stage 7 / 7 — Evaluating cluster quality")
    evaluator = Evaluator(config["evaluation"])
    metrics = evaluator.evaluate(reduced, labels)
    print(evaluator.interpret(metrics))

    banner("Exporting results")
    exporter = Exporter(config["output"])

    viz_path = None
    if config["output"].get("visualise", True):
        visualiser = Visualiser(config["output"])
        viz_path = visualiser.plot(viz_embeddings, labels, documents, cluster_info)
        logger.info("Interactive visualisation: %s", viz_path)

    export_paths = exporter.export(
        documents, labels, cluster_info, metrics, viz_path, config=config
    )

    elapsed = time.time() - t_start
    banner("Pipeline complete")
    print(f"  Total time : {elapsed:.1f}s")
    print(f"  Documents : {len(documents)}")
    print(f"  Clusters: {n_clusters}")
    print("\n  Output files:")
    for desc, path in export_paths.items():
        print(f"    {desc:<30} {path}")
    if viz_path:
        print(f"\n  Open in browser: {viz_path}")
    print()


if __name__ == "__main__":
    main()