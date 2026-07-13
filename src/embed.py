import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (OPENAI_API_KEY)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def format_repo_text(repo):
    """ Formats the repository JSON data into a single string """
    text_parts = [
        f"Repository: {repo.get('full_name', '')}",
        f"Description: {repo.get('description', '') or 'No description'}",
        f"README: \n{repo.get('readme', '') or 'No README'}"
    ]

    return "\n".join(text_parts)

def generate_embeddings(input_file="trending_repos.jsonl", output_file=None):
    """
    Reads repositories from a JSONL file, fetches embeddings from OpenAI, 
    and updates the file in place with an added 'embedding' key.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run ingest.py first.")
        return
        
    repos = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                repos.append(json.loads(line))
        
    print(f"Loaded {len(repos)} repositories from {input_file}.")
    
    repos_to_embed = [repo for repo in repos if not repo.get("embedding")]
    
    if len(repos_to_embed) < len(repos):
        print(f"Skipping {len(repos) - len(repos_to_embed)} repositories that already have embeddings.")
        
    batch_size = 50
    for i in range(0, len(repos_to_embed), batch_size):
        batch = repos_to_embed[i:i + batch_size]
        print(f"[{i+1} - {i+len(batch)} / {len(repos_to_embed)}] Generating embeddings for batch...")
        
        inputs = [format_repo_text(repo) for repo in batch]
        
        try:
            response = client.embeddings.create(
                input=inputs,
                model="text-embedding-3-small"
            )
            for j, item in enumerate(response.data):
                batch[j]["embedding"] = item.embedding
                
        except Exception as e:
            print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
            for repo in batch:
                repo["embedding"] = None

    save_path = output_file if output_file else input_file
    with open(save_path, "w", encoding="utf-8") as f:
        for repo in repos:
            f.write(json.dumps(repo, ensure_ascii=False) + '\n')
        
    print(f"\nDone! Appended embeddings to {save_path}")

if __name__ == "__main__":
    generate_embeddings()
