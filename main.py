from src.ingest import get_trending_repos
from src.embed import generate_embeddings
from src.cluster import cluster_repos
from src.tsne import visualize_clusters

if __name__ == "__main__":
    print("--- Step 1: Fetching Repositories ---")
    get_trending_repos(
        query="created:>2026-04-14 pushed:>2026-07-06 stars:100..20000",
        total_items=100,
        truncate_readme=2500,
        keep_archive=True
    )
    
    print("\n--- Step 2: Generating Embeddings ---")
    generate_embeddings()

    print("\n--- Step 3: Clustering Repositories ---")
    cluster_repos(min_cluster_size=2)
    
    print("\n--- Step 4: Visualizing Clusters ---")
    visualize_clusters(perplexity=None)
