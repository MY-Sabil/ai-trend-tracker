from src.ingest import get_trending_repos
from src.embed import generate_embeddings
from src.cluster import cluster_repos
from src.match import match_cluster_continuity
from src.label import run_semantic_labeling
from src.tsne import visualize_clusters
from src.export import export_trends

if __name__ == "__main__":
    print("--- Step 1: Fetching Repositories ---")
    get_trending_repos(
        query="created:>2026-01-01",
        total_items=700,
        truncate_readme=2000,
        keep_archive=True
    )
    
    print("\n--- Step 2: Generating Embeddings ---")
    generate_embeddings()

    print("\n--- Step 3: Clustering Repositories ---")
    cluster_repos(min_cluster_size=5, keep_archive=False)
    
    print("\n--- Step 4: Tracking Trend Continuity ---")
    match_cluster_continuity()
    
    print("\n--- Step 5: Semantic Labeling (Emergent Clusters) ---")
    run_semantic_labeling(sample_size=3, keep_archive=True)
    
    print("\n--- Step 6: Visualizing Clusters ---")
    visualize_clusters()
    
    print("\n--- Step 7: Exporting Final Data ---")
    export_trends()
