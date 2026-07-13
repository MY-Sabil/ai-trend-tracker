import os
import json
import numpy as np
from datetime import datetime
from sklearn.manifold import TSNE

def export_trends(repos_path="trending_repos.jsonl", history_path="clusters/history.jsonl", output_dir="trends", output_filename=None):
    """
    Consolidates the processed repository data and historical metrics into a structured
    JSON file intended for interactive front-end visualizations.
    """
    print("Generating consolidated export payload...")
    
    if not os.path.exists(repos_path):
        print(f"Error: {repos_path} not found.")
        return
        
    repos = []
    with open(repos_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                repos.append(json.loads(line))
                
    # 1. Compute t-SNE coordinates using same parameters as tsne.py for consistency
    embeddings = []
    valid_indices = []
    
    for i, repo in enumerate(repos):
        emb = repo.get("embedding")
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(i)
            
    if embeddings:
        X = np.array(embeddings)
        perplexity = min(30, max(2, len(X) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_2d = tsne.fit_transform(X)
        
        for idx, (x, y) in zip(valid_indices, X_2d):
            repos[idx]["x"] = round(float(x), 2)
            repos[idx]["y"] = round(float(y), 2)
            
    # 2. Load historical clusters to calculate velocity and acceleration metrics
    history_categories = {}
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cat = json.loads(line)
                    history_categories[cat.get("category_id")] = cat

    # 3. Build the categories array
    cat_groups = {}
    for repo in repos:
        c_id = repo.get("category_id")
        # Exclude noise points from the clean interactive data
        if not c_id or c_id == "noise":
            continue
            
        if c_id not in cat_groups:
            cat_groups[c_id] = {
                "label": repo.get("category_label", "Unknown"),
                "repos": [],
                "status": repo.get("trend_status", "continuous")
            }
        cat_groups[c_id]["repos"].append(repo)
        
    categories_output = []
    
    for c_id, group in cat_groups.items():
        repo_count = len(group["repos"])
        aggregate_stars = sum(r.get("stargazers_count", 0) for r in group["repos"])
        
        vel = 0
        acc = 0
        
        hist_cat = history_categories.get(c_id, {})
        metrics = hist_cat.get("metrics_history", [])
        
        # Calculate weekly growth velocity
        if len(metrics) >= 2:
            vel = metrics[-1].get("aggregate_stars", 0) - metrics[-2].get("aggregate_stars", 0)
            
        # Calculate weekly acceleration
        if len(metrics) >= 3:
            prev_vel = metrics[-2].get("aggregate_stars", 0) - metrics[-3].get("aggregate_stars", 0)
            acc = vel - prev_vel
            
        categories_output.append({
            "category_id": c_id,
            "label": group["label"],
            "repo_count": repo_count,
            "aggregate_stars": aggregate_stars,
            "star_velocity_weekly": vel,
            "star_acceleration_weekly": acc,
            "trend_status": group["status"]
        })
        
    # Sort categories by aggregate_stars descending
    categories_output.sort(key=lambda x: x["aggregate_stars"], reverse=True)
        
    # 4. Format repositories array
    repositories_output = []
    for repo in repos:
        c_id = repo.get("category_id")
        if not c_id or c_id == "noise":
            continue
            
        repositories_output.append({
            "full_name": repo.get("full_name"),
            "description": repo.get("description"),
            "stargazers_count": repo.get("stargazers_count"),
            "html_url": repo.get("html_url"),
            "primary_language": repo.get("language"),
            "category_id": c_id,
            "category_label": repo.get("category_label"),
            "x": repo.get("x", 0.0),
            "y": repo.get("y", 0.0)
        })
        
    # 5. Compile final JSON wrapper
    final_payload = {
        "generated_at": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        "total_repositories_scanned": len(repos),
        "categories": categories_output,
        "repositories": repositories_output
    }
    
    if output_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"trend-{timestamp}.json"
        
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)
        
    print(f"Exported interactive data to {out_path}")

if __name__ == "__main__":
    export_trends()
