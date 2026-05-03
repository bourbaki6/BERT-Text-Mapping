
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Exporter:
    

    def __init__(self, config: dict) -> None:
        self.results_dir = Path(config["results_dir"])
        self.save_cluster_csv: bool = config.get("save_cluster_csv", True)
        self.save_document_map: bool = config.get("save_document_map", True)
        self.save_report: bool = config.get("save_report", True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        documents: List[str],
        labels: np.ndarray,
        cluster_info: Dict[int, dict],
        metrics: Dict[str, float],
        viz_path: Optional[str] = None,
    ) -> Dict[str, str]:
        
        export_paths = {}


        if self.save_cluster_csv:
            path = self._save_cluster_csv(cluster_info)
            export_paths["cluster_keywords_csv"] = str(path)

        if self.save_document_map:
            path = self._save_document_map(documents, labels, cluster_info)
            export_paths["document_map_csv"] = str(path)

        path = self._save_metrics(metrics)
        export_paths["metrics_json"] = str(path)

        
        if self.save_report:
            path = self._save_report(cluster_info, metrics, viz_path)
            export_paths["text_report"] = str(path)

    
        path = self._save_cluster_details(cluster_info)
        export_paths["cluster_details_dir"] = str(path)

        return export_paths



    def _save_cluster_csv(self, cluster_info: Dict[int, dict]) -> Path:
       
        import csv
       
        path = self.results_dir / "cluster_keywords.csv"
        with open(path, "w", newline = "", encoding = "utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["cluster_id", "label", "size", "keywords"])
            for cid in sorted(cluster_info.keys()):
                info = cluster_info[cid]
                writer.writerow([
                    cid,
                    info["label"],
                    info["size"],
                    " | ".join(info.get("keywords", [])),
                ])
        logger.debug("Saved cluster keywords CSV: %s", path)
        return path

    def _save_document_map(self,documents: List[str],labels: np.ndarray, cluster_info: Dict[int, dict]) -> Path:
        
        import csv
        path = self.results_dir / "document_map.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
            writer.writerow(["doc_id", "cluster_id", "cluster_label", "document"])
            for i, (doc, label) in enumerate(zip(documents, labels)):
                cid = int(label)
                cluster_label = cluster_info.get(cid, {}).get("label", "Noise/Unknown")
                
                writer.writerow([i, cid, cluster_label, doc[:300]])
        logger.debug("Saved document map CSV: %s", path)
        return path

    def _save_metrics(self, metrics: Dict[str, float]) -> Path:
        
        path = self.results_dir / "metrics.json"

        clean_metrics = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in metrics.items()
        }
        with open(path, "w", encoding = "utf-8") as fh:
            json.dump(clean_metrics, fh, indent = 2)
        logger.debug("Saved metrics: %s", path)
        return path

    def _save_report(self, cluster_info: Dict[int, dict], metrics: Dict[str, float],viz_path: Optional[str]) -> Path:
       
        path = self.results_dir / "cluster_summary.txt"
        lines = []

        lines.append("  BERT TEXT CLUSTERING — RESULTS SUMMARY")
        
        lines.append("")

    
        lines.append("EVALUATION METRICS")
     
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:<35} {v:.4f}")
            else:
                lines.append(f"  {k:<35} {v}")
        lines.append("")

    
        n_clusters = sum(1 for cid in cluster_info if cid != -1)
        total_docs = sum(info["size"] for info in cluster_info.values())
        lines.append(f"CLUSTERS FOUND: {n_clusters}")
        lines.append(f"TOTAL DOCUMENTS: {total_docs}")
        lines.append("")

        for cid in sorted(cluster_info.keys()):
            info = cluster_info[cid]
            cluster_type = "NOISE" if cid == -1 else f"CLUSTER {cid}"
            lines.append(f"{cluster_type}: {info['label']}")
            lines.append(f"  Size : {info['size']} documents")
            lines.append(f" Keywords : {', '.join(info.get('keywords', []))}")
            lines.append("Representative documents:")
            for doc in info.get("docs", [])[:3]:
                # 
                wrapped = self._wrap_text(doc[:160], width=65, indent="    ")
                lines.append(wrapped)
            lines.append("")
        
    
        if viz_path:
           
            lines.append(f"Interactive visualisation: {viz_path}")
            lines.append("Open in any web browser to explore clusters interactively.")

        lines.append("")

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        logger.debug("Saved text report: %s", path)
        return path

    def _save_cluster_details(self, cluster_info: Dict[int, dict]) -> Path:
        
        detail_dir = self.results_dir / "cluster_details"
        detail_dir.mkdir(exist_ok = True)

        for cid, info in cluster_info.items():
            fname = f"cluster_{cid:03d}.json" if cid != -1 else "cluster_noise.json"
            path = detail_dir / fname
            
            serialisable = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in info.items()
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(serialisable, fh, indent=2, ensure_ascii=False)

        logger.debug("Saved %d cluster detail files to: %s", len(cluster_info), detail_dir)
        return detail_dir

    @staticmethod
    def _wrap_text(text: str, width: int = 65, indent: str = "    ") -> str:
        
        words = text.split()
        lines = []
        current = indent
        for word in words:
            if len(current) + len(word) + 1 > width:
                lines.append(current.rstrip())
                current = indent + word + " "
            else:
                current += word + " "
        if current.strip():
            lines.append(current.rstrip())
        return "\n".join(lines)

if __name__ == "__main__":
    import sys, json, pathlib, time
    sys.path.insert(0, ".")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_norm
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from data.dataset import DataLoader
    from labelling.labels import ClusterLabeller
    from evaluation.eval import Evaluator

    print("=" * 60)
    print("  Exporter — Stage 8: Saving Results to Disk")
    print("=" * 60)

    
    loader = DataLoader({"input_path": "demo", "min_doc_length": 20, "strip_html": True})
    docs = loader.load()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2, sublinear_tf=True)
    embs = sk_norm(vec.fit_transform(docs).toarray(), norm="l2").astype("float32")
    reduced = TruncatedSVD(n_components=10, random_state=42).fit_transform(embs).astype("float32")
    labels = KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(reduced).astype(int)
    labeller = ClusterLabeller(
        {"method":"tfidf","top_n_keywords":8,"use_ngrams":True,"ngram_range":[1,2]}, None)
    cluster_info = labeller.label(docs, labels, embs)
    metrics_dict = Evaluator({"silhouette":True,"davies_bouldin":True,
                               "calinski_harabasz":True,"max_sample_for_metrics":5000}
                              ).evaluate(reduced, labels)


    config = {"results_dir":"results","save_cluster_csv":True,
              "save_document_map":True,"save_report":True}
    pathlib.Path("results").mkdir(exist_ok=True)

    t0 = time.time()
    exporter = Exporter(config)
    paths = exporter.export(docs, labels, cluster_info, metrics_dict)
    elapsed = time.time() - t0

    print(f"\n  Export time : {elapsed:.3f}s")
    print(f"\n  Files written:")
    for desc, path in paths.items():
        size = pathlib.Path(path).stat().st_size if pathlib.Path(path).is_file() else 0
        size_str = f"{size:,} bytes" if size else "directory"
        print(f" {desc:<25} -> {path}  ({size_str})")

    
    import csv
    csv_path = pathlib.Path("results/cluster_keywords.csv")
    if csv_path.exists():
        print(f"\n  Preview: cluster_keywords.csv")
       
        with open(csv_path) as f:
            for i, line in enumerate(f):
                if i > 12: print("  ..."); break
                print(f"  {line.rstrip()}")


    json_path = pathlib.Path("results/metrics.json")
    if json_path.exists():
        print(f"\n  Preview: metrics.json")
        
        with open(json_path) as f:
            for line in f:
                print(f"  {line.rstrip()}")

    report_path = pathlib.Path("results/cluster_summary.txt")
    if report_path.exists():
        print(f"\n  Preview: cluster_summary.txt (first 30 lines)")
    
        lines = report_path.read_text().split("\n")
        for line in lines[:30]:
            print(f"  {line}")

    print("\n  Exporter OK — full pipeline complete!")
    print("  Run main.py to execute all stages together.")