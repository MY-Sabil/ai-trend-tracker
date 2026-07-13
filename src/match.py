import os
import json
import uuid

def calculate_set_metrics(set_a, set_b):
    """
    Computes Jaccard similarity between two sets of string tokens.
    """
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    # Avoid zero division
    jaccard_index = intersection / union if union > 0 else 0.0
    return jaccard_index

def match_cluster_continuity(current_path="trending_repos.jsonl", history_path="clusters/history.jsonl", output_path=None, jaccard_threshold=0.3):
    """
    Matches current clusters against historical clusters to ensure trend continuity across runs.
    """
    # 1. Load the historical categories state
    historical_categories = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    historical_categories.append(json.loads(line))
    else:
        print("No historical context file located.")

    # 2. Load the current execution's clustered repositories
    if not os.path.exists(current_path):
        print(f"Error: {current_path} not found.")
        return
        
    current_repos = []
    with open(current_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                current_repos.append(json.loads(line))

    # 3. Group current repos into active text sets based on temporary HDBSCAN cluster IDs
    new_clusters = {}
    for repo in current_repos:
        c_id = repo.get("cluster_id", -1)
        # Skip noise markers (-1)
        if c_id == -1:
            continue
        if c_id not in new_clusters:
            new_clusters[c_id] = set()
        new_clusters[c_id].add(repo["full_name"])

    # 4. Process the intersection evaluation matrix
    temp_id_to_stable_metadata = {}

    for temp_id, new_repo_set in new_clusters.items():
        best_jaccard = 0.0
        matched_history_category = None

        for hist_cat in historical_categories:
            hist_repo_set = set(hist_cat["repository_names"])
            jaccard = calculate_set_metrics(new_repo_set, hist_repo_set)

            # Keep track of the historical category with the tightest fit
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                matched_history_category = hist_cat

        # 5. Apply threshold routing rules
        if best_jaccard >= jaccard_threshold and matched_history_category is not None:
            print(f"Match Found: Temporary Cluster #{temp_id} matches historical trend "
                  f"'{matched_history_category['label']}' (Jaccard: {best_jaccard:.2f})")
            
            temp_id_to_stable_metadata[temp_id] = {
                "category_id": matched_history_category["category_id"],
                "label": matched_history_category["label"],
                "status": "continuous"
            }
        else:
            new_uuid = str(uuid.uuid4())
            print(f"Emergent Trend: Temporary Cluster #{temp_id} shows no historical alignment. "
                  f"Allocating new Profile: {new_uuid[:8]}")
            
            temp_id_to_stable_metadata[temp_id] = {
                "category_id": new_uuid,
                "label": f"Trend {new_uuid[:8]}",
                "status": "emergent"
            }

    # 6. Inject the resolved tracking IDs into the active repositories list
    for repo in current_repos:
        c_id = repo.get("cluster_id", -1)
        if c_id == -1:
            repo["category_id"] = "noise"
            repo["category_label"] = "Background Static"
            repo["trend_status"] = "ignored"
        else:
            meta = temp_id_to_stable_metadata[c_id]
            repo["category_id"] = meta["category_id"]
            repo["category_label"] = meta["label"]
            repo["trend_status"] = meta["status"]

    # 7. Update Historical Memory
    from datetime import datetime
    now = datetime.utcnow()
    current_time_str = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    # Use %V for ISO week number
    current_week_str = now.strftime('%Y-W%V')

    final_history_rows = []
    matched_cat_ids = set([meta["category_id"] for meta in temp_id_to_stable_metadata.values()])
    
    # Retain old trends that weren't matched this time so we don't forget them
    for hist_cat in historical_categories:
        if hist_cat.get("category_id") not in matched_cat_ids:
            final_history_rows.append(hist_cat)
            
    # Add/Update the actively matched and emergent trends
    for c_id, metadata in temp_id_to_stable_metadata.items():
        cat_id = metadata["category_id"]
        
        # Calculate current metrics for this cluster
        cluster_repos_list = [r for r in current_repos if r.get("cluster_id") == c_id]
        repo_count = len(cluster_repos_list)
        aggregate_stars = sum(r.get("stargazers_count", 0) for r in cluster_repos_list)
        
        current_metric = {
            "week": current_week_str,
            "repo_count": repo_count,
            "aggregate_stars": aggregate_stars
        }
        
        if metadata["status"] == "continuous":
            # Update the existing historical record
            old_cat = next((cat for cat in historical_categories if cat.get("category_id") == cat_id), None)
            if old_cat:
                old_cat["last_active"] = current_time_str
                old_cat["repository_names"] = list(new_clusters[c_id])
                
                if "metrics_history" not in old_cat:
                    old_cat["metrics_history"] = []
                    
                # If we run multiple times in the same week, update the last entry instead of appending duplicates
                if old_cat["metrics_history"] and old_cat["metrics_history"][-1].get("week") == current_week_str:
                    old_cat["metrics_history"][-1] = current_metric
                else:
                    old_cat["metrics_history"].append(current_metric)
                    
                final_history_rows.append(old_cat)
        else:
            # Create a brand new emergent trend profile
            new_cat = {
                "category_id": cat_id,
                "label": metadata["label"],
                "first_detected": current_time_str,
                "last_active": current_time_str,
                "repository_names": list(new_clusters[c_id]),
                "metrics_history": [current_metric]
            }
            final_history_rows.append(new_cat)

    # Ensure the target directory for the history file exists (e.g. 'clusters/' folder)
    history_dir = os.path.dirname(history_path)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)

    with open(history_path, "w", encoding="utf-8") as f:
        for row in final_history_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 8. Save the synchronized repositories snapshot list
    save_path = output_path if output_path else current_path
    with open(save_path, "w", encoding="utf-8") as f:
        for repo in current_repos:
            f.write(json.dumps(repo, ensure_ascii=False) + "\n")

    print(f"\nIdentity matching execution successfully completed. Matrix mappings saved to: {save_path}")

if __name__ == "__main__":
    match_cluster_continuity()
