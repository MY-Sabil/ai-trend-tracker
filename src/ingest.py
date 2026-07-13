import os
import re
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables (GITHUB_TOKEN)
load_dotenv()

def get_readme(owner, repo, headers):
    """Fetches the raw README.md text for a given repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    readme_headers = headers.copy()
    readme_headers["Accept"] = "application/vnd.github.v3.raw"
    
    response = requests.get(url, headers=readme_headers)
    if response.status_code == 200:
        return response.text
    return None

def archive_results(retained_data, query):
    """Saves a copy of the results to the archive folder with a timestamp."""
    archive_dir = os.path.join("archive", "repos")
    os.makedirs(archive_dir, exist_ok=True)
    
    sanitized_query = re.sub(r'[^a-zA-Z0-9]', '_', query).strip('_')
    sanitized_query = re.sub(r'_+', '_', sanitized_query)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_filename = f"{sanitized_query}_{timestamp}.jsonl"
    archive_path = os.path.join(archive_dir, archive_filename)
    
    with open(archive_path, "w", encoding="utf-8") as f:
        for repo in retained_data:
            f.write(json.dumps(repo, ensure_ascii=False) + '\n')
        
    print(f"Archived copy saved to {archive_path}")

def get_trending_repos(query=None, sort="stars", order="desc", total_items=5, output_file="trending_repos.jsonl", truncate_readme=1500, keep_archive=False):
    """
    Fetches trending repositories from GitHub API and saves their metadata and READMEs in a JSONL file.
    Paginates automatically if total_items > 100.
    """
    # Default to fetching repositories created in the last 3 months and pushed in the last week
    if query is None:
        three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        last_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        query = f"created:>{three_months_ago} pushed:>{last_week}"

    url = "https://api.github.com/search/repositories"
    
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    retained_data = []
    page = 1
    
    print(f"Fetching {total_items} repositories with query: {query}")
    
    try:
        while len(retained_data) < total_items:
            current_per_page = min(100, total_items - len(retained_data))
            
            params = {
                "q": query,
                "sort": sort,
                "order": order,
                "per_page": current_per_page,
                "page": page
            }
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            repos = data.get("items", [])
            
            if not repos:
                break
            
            for repo in repos:
                if len(retained_data) >= total_items:
                    break
                    
                owner = repo.get("owner", {})
                owner_login = owner.get("login")
                repo_name = repo.get("name")
                
                print(f"[{len(retained_data)+1}/{total_items}] Processing: {repo.get('full_name')}")
                
                # Fetch and optionally truncate the README
                readme_text = get_readme(owner_login, repo_name, headers)
                if readme_text and truncate_readme and truncate_readme > 0:
                    readme_text = readme_text[:truncate_readme]
                
                repo_data = {
                    "name": repo_name,
                    "full_name": repo.get("full_name"),
                    "owner_login": owner_login,
                    "owner_type": owner.get("type"),
                    "owner_html_url": owner.get("html_url"),
                    "html_url": repo.get("html_url"),
                    "description": repo.get("description"),
                    "stargazers_count": repo.get("stargazers_count"),
                    "language": repo.get("language"),
                    "topics": repo.get("topics", []),
                    "created_at": repo.get("created_at"),
                    "readme": readme_text
                }
                
                retained_data.append(repo_data)
                
            page += 1
            
        # Write collected data to local JSONL file
        with open(output_file, "w", encoding="utf-8") as f:
            for repo in retained_data:
                f.write(json.dumps(repo, ensure_ascii=False) + '\n')
            
        print(f"\nSuccessfully saved {len(retained_data)} repositories to {output_file}")
        
        # Archive copy if requested
        if keep_archive:
            archive_results(retained_data, query)
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from GitHub API: {e}")

if __name__ == "__main__":
    get_trending_repos(
        query=None, # Leave as None to use the default 'last 3 months / pushed last week' query
        sort="stars", 
        order="desc", 
        total_items=5, 
        output_file="trending_repos.jsonl", 
        truncate_readme=1500
    )
