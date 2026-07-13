import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_title_from_llm(repo_samples):
    """
    Sends a curated list of top repositories to OpenAI to generate a clean category name.
    """
    # Format the input data cleanly for the model
    context_lines = []
    for repo in repo_samples:
        topics_str = ", ".join(repo.get("topics", []) or [])
        description = repo.get("description") or "No description provided."
        readme = repo.get("readme") or "No README provided."
        
        line = (
            f"Name: {repo.get('full_name')}\n"
            f"Description: {description}\n"
            f"Tags: {topics_str}\n"
            f"README Excerpt:\n{readme}\n"
            "---"
        )
        context_lines.append(line)
    
    formatted_repos = "\n".join(context_lines)
    
    system_instruction = (
        "You are an expert technical analyst specializing in open-source AI software. "
        "Your task is to review a small group of related GitHub repositories and identify "
        "the precise common technical theme or methodology that unites them."
    )
    
    user_prompt = (
        "Analyze these repositories and provide a professional category title that is "
        "between 2 and 5 words long (e.g., 'Vector Search Engines' or 'Model Quantization Tools').\n\n"
        "Rules:\n"
        "1. Return ONLY the raw text string of the title.\n"
        "2. Do not include quotes, markdown formatting, explanations, or introductory filler phrase text.\n\n"
        f"Repositories:\n{formatted_repos}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=15
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error encountered during labeling: {e}")
        return None

def run_semantic_labeling(current_path="trending_repos.jsonl", history_path="clusters/history.jsonl", sample_size=3, keep_archive=True):
    """
    Finds newly emergent clusters and generates human-readable semantic labels using an LLM.
    Updates both the active snapshot and the historical memory.
    """
    # 1. Load current repositories snapshot
    if not os.path.exists(current_path):
        print(f"Error: {current_path} not found.")
        return
        
    current_repos = []
    with open(current_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                current_repos.append(json.loads(line))
            
    # 2. Group only the emergent repositories by their stable category identity
    emergent_groups = {}
    for repo in current_repos:
        if repo.get("trend_status") == "emergent":
            cat_id = repo["category_id"]
            if cat_id not in emergent_groups:
                emergent_groups[cat_id] = []
            emergent_groups[cat_id].append(repo)
            
    if not emergent_groups:
        print("No brand-new emergent trends found this run. Skipping LLM labeling phase.")
        return

    print(f"Found {len(emergent_groups)} new emergent categories requiring titles.")
    
    # 3. Process each emergent group to discover its human title
    resolved_titles = {}
    for cat_id, repos in emergent_groups.items():
        # Sort by stars descending to pick the most influential projects
        sorted_repos = sorted(repos, key=lambda x: x.get("stargazers_count", 0), reverse=True)
        top_samples = sorted_repos[:sample_size]
        
        print(f"Requesting name for category profile {cat_id[:8]} using top contributors...")
        generated_title = generate_title_from_llm(top_samples)
        
        if generated_title:
            print(f"-> Successfully assigned title: '{generated_title}'")
            resolved_titles[cat_id] = generated_title
        else:
            # Fallback if API acts up
            resolved_titles[cat_id] = f"Emergent Trend ({cat_id[:8]})"

    # 4. Overwrite placeholder names inside the active repository file
    for repo in current_repos:
        cat_id = repo.get("category_id")
        if cat_id in resolved_titles:
            repo["category_label"] = resolved_titles[cat_id]
            repo["trend_status"] = "emergent-labeled" # Update status to indicate processing is done

    with open(current_path, "w", encoding="utf-8") as f:
        for repo in current_repos:
            f.write(json.dumps(repo, ensure_ascii=False) + "\n")

    # 5. Synchronize and update the labels inside the historical archive ledger file
    if os.path.exists(history_path):
        historical_rows = []
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cat_data = json.loads(line)
                    cat_id = cat_data.get("category_id")
                    if cat_id in resolved_titles:
                        cat_data["label"] = resolved_titles[cat_id]
                    historical_rows.append(cat_data)
                
        with open(history_path, "w", encoding="utf-8") as f:
            for row in historical_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Step 5 complete. Cluster data storage files successfully updated with LLM semantic tags.")

    if keep_archive:
        from datetime import datetime
        archive_dir = os.path.join("archive", "final")
        os.makedirs(archive_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_filename = f"{len(current_repos)}_repos_{timestamp}.jsonl"
        archive_path = os.path.join(archive_dir, archive_filename)
        
        with open(archive_path, "w", encoding="utf-8") as f:
            for repo in current_repos:
                f.write(json.dumps(repo, ensure_ascii=False) + '\n')
                
        print(f"Archived labeled copy saved to {archive_path}")

if __name__ == "__main__":
    run_semantic_labeling()
