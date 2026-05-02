
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


CLUSTER_COLOURS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]
NOISE_COLOUR = "#CCCCCC"  


class Visualiser:
   

    def __init__(self, config: dict) -> None:
        self.results_dir = Path(config["results_dir"])

   
    def plot(
        self,
        viz_embeddings: np.ndarray,
        labels: np.ndarray,
        documents: List[str],
        cluster_info: Dict[int, dict],
    ) -> str:
       
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error(
                "plotly is not installed. Run: pip install plotly"
            )
            return ""

        fig = go.Figure()

        unique_labels = sorted(set(labels))

        for cid in unique_labels:
            mask = labels == cid
            x = viz_embeddings[mask, 0]
            y = viz_embeddings[mask, 1]

            if cid == -1:
                
                name = "Noise"
                colour = NOISE_COLOUR
                keywords_str = "Unassigned documents"
            
            else:
                info = cluster_info.get(cid, {})
                name = f"[{cid}] {info.get('label', f'Cluster {cid}')}"
                colour = CLUSTER_COLOURS[cid % len(CLUSTER_COLOURS)]
                kws = info.get("keywords", [])[:5]
                keywords_str = ", ".join(kws)

    
            hover_texts = [
                f"<b>{name}</b><br>"
                f"<i>Keywords: {keywords_str}</i><br>"
                f"─────────────────────<br>"
                f"{doc[:200]}{'…' if len(doc) > 200 else ''}"
                for doc in [documents[i] for i in np.where(mask)[0]]
            ]

            fig.add_trace(
                go.Scattergl(
                    x = x,
                    y = y,
                    mode = "markers",
                    name = name,
                    marker = dict(
                        color = colour,
                        size = 6,
                        opacity = 0.7 if cid != -1 else 0.3,
                        line = dict(width = 0.3, color ="white"),
                    ),
                    text = hover_texts,
                    hoverinfo = "text",
                )
            )

        
        fig.update_layout(
            title=dict(
                text = "BERT Text Clustering — 2D UMAP Projection",
                font = dict(size = 18),
                x = 0.5,
            ),
            
            xaxis = dict(title = "UMAP Dimension 1", showgrid = False, zeroline = False),
            yaxis = dict(title = "UMAP Dimension 2", showgrid = False, zeroline = False),
            
            legend = dict(
                title = dict(text = "Clusters"),
                itemsizing = "constant",
                borderwidth = 1,
            ),
            hovermode = "closest",
            
            plot_bgcolor = "white",
            paper_bgcolor =" white",
            
            width = 1100,
            height = 750,
            
            margin = dict(l = 60, r = 60, t = 80, b = 60),
        )

       
        output_path = self.results_dir / "cluster_viz.html"
        fig.write_html(
            str(output_path),
            include_plotlyjs = True,  
            full_html = True,
        )
        logger.info("Saved interactive plot: %s", output_path)
        
        return str(output_path)
    
