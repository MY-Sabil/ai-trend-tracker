import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_clusters(input_file="trending_repos.jsonl", output_image="clusters.png", perplexity=None):
    """
    Reads the repository data (with embeddings and cluster IDs),
    runs t-SNE to reduce the embeddings to 2D, and saves a scatter plot.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run cluster.py first.")
        return
        
    repos = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                repos.append(json.loads(line))
                
    embeddings = []
    labels = []
    names = []
    
    for repo in repos:
        emb = repo.get("embedding")
        cluster_id = repo.get("cluster_id", -1) 
        
        if emb is not None:
            embeddings.append(emb)
            labels.append(cluster_id)
            names.append(repo.get("name"))
            
    if not embeddings:
        print("No embeddings found to visualize.")
        return
        
    X = np.array(embeddings)
    n_samples = len(X)
    
    if perplexity is None:
        perplexity = min(30, max(2, n_samples - 1))
    
    print(f"Running t-SNE on {n_samples} items (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    
    category_names = {}
    for repo in repos:
        c_id = repo.get("cluster_id", -1)
        if c_id != -1 and c_id not in category_names:
            category_names[c_id] = repo.get("category_label", f"Cluster {c_id}")

    unique_labels = set(labels)
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue
            
        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        cluster_points = X_2d[indices]
        
        label_name = category_names.get(cluster_id, f'Cluster {cluster_id}')
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1], 
            s=100, label=label_name
        )
            
    for i, name in enumerate(names):
        if labels[i] == -1:
            continue
            
        plt.annotate(
            name, 
            (X_2d[i, 0], X_2d[i, 1]), 
            xytext=(5, 5), 
            textcoords='offset points', 
            fontsize=9, 
            alpha=0.7
        )
        
    plt.title("t-SNE Visualization of Repository Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Visualization saved successfully to {output_image}")

if __name__ == "__main__":
    visualize_clusters()
