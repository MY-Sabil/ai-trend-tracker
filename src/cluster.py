import os
import json
import numpy as np
from datetime import datetime
from sklearn.cluster import HDBSCAN

def cluster_repos(input_file="trending_repos.jsonl", output_file=None, min_cluster_size=2, keep_archive=False):
    """
    Reads repositories with embeddings from a JSONL file,
    clusters them using HDBSCAN, and updates the file with a 'cluster_id' key.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run embed.py first.")
        return

    repos = []
    embeddings = []
    
    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                repo = json.loads(line)
                repos.append(repo)
                embeddings.append(repo.get("embedding"))
    
    print(f"Loaded {len(embeddings)} repositories.")
    
    valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
    
    if not valid_indices:
        print("No valid embeddings found to cluster.")
        return
        
    valid_embeddings = [embeddings[i] for i in valid_indices]
    
    X = np.array(valid_embeddings)
    
    print(f"Clustering {len(valid_indices)} embeddings using HDBSCAN...")
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, copy=True)
    labels = hdb.fit_predict(X)
    
    for idx, label in zip(valid_indices, labels):
        repos[idx]["cluster_id"] = int(label)
        
    for i in range(len(repos)):
        if "cluster_id" not in repos[i]:
            repos[i]["cluster_id"] = -1
            
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = list(labels).count(-1)
    print(f"Found {num_clusters} distinct clusters (and {noise_count} noise points).")
    
    save_path = output_file if output_file else input_file
    with open(save_path, "w", encoding="utf-8") as f:
        for repo in repos:
            f.write(json.dumps(repo, ensure_ascii=False) + '\n')
            
    print(f"\nDone! Appended cluster IDs to {save_path}")

    if keep_archive:
        archive_dir = os.path.join("archive", "clustered")
        os.makedirs(archive_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_filename = f"{len(repos)}_repos_{timestamp}.jsonl"
        archive_path = os.path.join(archive_dir, archive_filename)
        
        with open(archive_path, "w", encoding="utf-8") as f:
            for repo in repos:
                f.write(json.dumps(repo, ensure_ascii=False) + '\n')
                
        print(f"Archived final copy saved to {archive_path}")

if __name__ == "__main__":
    cluster_repos()
