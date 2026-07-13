# GitHub AI Trend Tracker

**GitHub AI Trend Tracker** tracks the shape of the AI world through raw GitHub activity. Instead of relying on hype threads or news aggregators, it uses what developers are *actually building* as ground truth. New techniques, frameworks, and ideas show up as repositories before they show up anywhere else. This project turns that raw signal into a readable, ongoing trend map.

It answers the question: *"What's actually heating up in AI right now, based on what people are building?"*

By letting clusters emerge organically rather than forcing repositories into hardcoded categories (like "LLMs" or "computer vision"), the tracker discovers brand-new AI categories as soon as they appear in the wild.

## Features & Pipeline
The pipeline runs through a robust 7-step process orchestrated by `main.py`:

1. **Ingestion (`src/ingest.py`)**: Fetches trending AI-related repositories using the GitHub Search API. Avoids brittle web scraping, automatically paginates, and pulls down READMEs, descriptions, and metadata.
2. **Embeddings (`src/embed.py`)**: Uses OpenAI's `text-embedding-3-small` model to generate high-quality vector embeddings of each repository's text. Requests are heavily batched for speed and to avoid API rate limits.
3. **Clustering (`src/cluster.py`)**: First reduces the dimensionality of the embeddings using **UMAP**, then groups the repositories into emergent categories using **HDBSCAN**. This allows us to discover natural clusters without needing to pre-define how many clusters exist, while smartly isolating noise.
4. **Trend Continuity (`src/match.py`)**: Solves the "cluster continuity problem" by using Jaccard similarity to match this week's clusters against historical data. This lets us track a cluster's growth, velocity, and acceleration over time.
5. **Semantic Labeling (`src/label.py`)**: Automatically assigns human-readable titles (e.g., "AI Agent Orchestration Frameworks" or "Local Inference Runtimes") to brand-new emergent clusters using `gpt-4o-mini`.
6. **Visualization (`src/tsne.py`)**: Projects the high-dimensional embeddings down to 2D using **t-SNE** and generates a clean, labeled scatter plot (`clusters.png`) of the current AI landscape.
7. **Export (`src/export.py`)**: Compiles all the processed data, historical metrics, and coordinates into a pristine JSON payload (`trends/trend-[timestamp].json`), ready to be plugged into an interactive front-end dashboard.

## Setup & Usage

### Prerequisites
1. Clone the repository.
2. Install the required Python packages (e.g., in a `.venv`):
   ```bash
   pip install requests openai scikit-learn numpy matplotlib python-dotenv umap-learn
   ```
3. Create a `.env` file in the root directory with your API keys:
   ```env
   GITHUB_TOKEN=your_github_personal_access_token
   OPENAI_API_KEY=your_openai_api_key
   ```

### Running the Pipeline
Simply run the main orchestrator script:
```bash
python main.py
```

This will run all 7 steps sequentially, generate visual plots, track history in `clusters/history.jsonl`, and output the final dashboard payload into the `trends/` folder.

## Future Roadmap / Extra Signals
- **Velocity vs Volume**: Focus heavily on rate of change rather than raw volume.
- **Language/Stack Shifts**: Track if a trending category is mostly Python or shifting toward Rust/Go.
- **Frontend Dashboard**: A deployable Next.js/Vite frontend to consume the exported JSON payload interactively.
- **Weekly Digest**: An auto-generated summary paragraph of "what moved this week," acting as an autonomous newsletter.
