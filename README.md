# Custom Vector Database & AI Assistant

**Built by Arvind Dwivedi** · Python, NLP, Vector Search

---

- Developed a custom vector database to store and retrieve text embeddings using cosine similarity for semantic search.
- Implemented an AI-powered Retrieval-Augmented Generation (RAG) pipeline with document indexing, embedding generation, and context-aware response generation.
- Optimized embedding storage and similarity search for scalable AI applications and efficient information retrieval.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)                ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving O(log N) complexity instead of O(N) for brute force.

---

## Prerequisites

You need **2 things** installed:

1. **Python 3.8+** (with pip)
2. **Ollama** (runs the local AI models for document embedding and RAG)

---

## Step-by-Step Setup

### Step 1 — Install Python

If you don't have Python installed:

- **Windows:** Download from [python.org](https://www.python.org/downloads/) — check "Add Python to PATH" during install
- **macOS:** `brew install python`
- **Linux:** `sudo apt install python3 python3-pip`

Verify:
```bash
python --version
```

---

### Step 2 — Install Ollama (Local AI Models)

1. Go to **https://ollama.com** and download for your OS
2. Run the installer
3. Pull the two required models:

```bash
ollama pull nomic-embed-text
```
*(~274 MB — this is the embedding model)*

```bash
ollama pull llama3.2
```
*(~2 GB — this is the language model for RAG)*

4. Verify:
```bash
ollama list
```

> **Minimum specs:** 8GB RAM recommended. The models use ~3GB total.

---

### Step 3 — Clone and Install

```bash
git clone https://github.com/ArvindDwivedi/Custom-VectorDB-AI-Assistant.git
cd Custom-VectorDB-AI-Assistant

pip install -r requirements.txt
```

---

### Step 4 — Run

**Terminal 1** — Start Ollama (if not already running):
```bash
ollama serve
```

**Terminal 2** — Start the server:
```bash
python app.py
```

You should see:
```
=======================================================
  Custom Vector Database & AI Assistant
  Built by Arvind Dwivedi
=======================================================
  Server:     http://localhost:8080
  Demo:       20 vectors | 16 dims | HNSW+KD-Tree+BruteForce
  Ollama:     ONLINE
  Embed:      nomic-embed-text
  Generate:   llama3.2
=======================================================
```

**Open your browser** → `http://localhost:8080`

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Type any concept: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, matching points glow on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to benchmark all 3 algorithms head-to-head

**The scatter plot** shows all 20 vectors projected to 2D using PCA. The 4 semantic categories (CS, Math, Food, Sports) form distinct clusters — this is what "semantic similarity" looks like visually.

### Tab 2: Documents (Real Embeddings)

Uses Ollama to generate **real 768-dimensional embeddings** from any text.

1. Type a title (e.g., `Operating Systems Notes`)
2. Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks
5. Each chunk gets its own embedding and is stored in the HNSW index

### Tab 3: Ask AI (RAG Pipeline)

1. Insert some documents in Tab 2 first
2. Type a question about your documents
3. Click **🤖 ASK AI**

What happens behind the scenes:
```
1. Your question → embedded with nomic-embed-text (768D vector)
2. HNSW search → finds 3 most semantically similar chunks
3. Retrieved chunks → sent as context to llama3.2
4. llama3.2 → generates an answer based on your documents
```

---

## REST API Reference

The server exposes a full REST API at `http://localhost:8080`.

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

### Example: Search via curl

```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Ask a question via curl

```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Project Structure

```
Custom-VectorDB-AI-Assistant/
├── app.py              ← Python backend (HNSW, KD-Tree, BruteForce, REST API, RAG)
├── index.html          ← Frontend (PCA scatter plot, chat UI, benchmark)
├── requirements.txt    ← Python dependencies (Flask, NumPy, Requests)
└── README.md           ← This file
```

### Architecture (app.py)

```
BruteForce          O(N·d)      Exact, baseline
KDTree              O(log N)    Exact, axis-aligned partitioning
HNSW                O(log N)    Approximate, multilayer small-world graph

VectorDB            Unified interface over all 3 (16D demo vectors)
DocumentDB          HNSW-only index for real Ollama embeddings (768D)
OllamaClient        HTTP client → /api/embeddings + /api/generate
```

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

Nodes are inserted into a multilayer graph. Each node randomly gets assigned a maximum layer. Layer 0 has all nodes with many connections; higher layers have fewer nodes (exponentially fewer) with longer-range connections.

**Insert:** Start at the top layer, greedily find the nearest node, drop a layer, repeat. At each layer from your assigned max down to 0, run a beam search (ef_construction=200) and connect to the M nearest neighbors bidirectionally.

**Search:** Same greedy descent from top layer. At layer 0, expand to ef nearest candidates using a priority queue.

**Why it's fast:** The upper layers act like a highway — you quickly get to the right neighborhood, then zoom in at layer 0.

### KD-Tree (K-Dimensional Tree)

Binary space partitioning. Each node splits space along one dimension (cycling through all dimensions). Search prunes entire subtrees when the closest possible point in that subtree can't beat the current best.

**Weakness:** Degrades with high dimensions (curse of dimensionality). Works well for ≤20D, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions, almost all the space is near the boundary of the hypersphere — no subtrees get pruned. HNSW's graph-based approach doesn't have this problem.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Backend server and all search algorithms |
| **Flask** | REST API framework |
| **NumPy** | Optimized vector operations and distance calculations |
| **Ollama** | Local LLM inference (embedding + generation) |
| **HTML/CSS/JS** | Frontend with PCA visualization |

---

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in header | Run `ollama serve` in a terminal |
| Embedding takes forever | Ollama is downloading the model on first use, wait 2 min |
| Port 8080 already in use | `lsof -i :8080` then `kill <pid>` (or `netstat -ano \| findstr 8080` on Windows) |
| LLM answer is slow | Normal — llama3.2 takes 10–30s on CPU. Use `llama3.2:1b` for faster answers |

### Use a Smaller/Faster LLM

```bash
ollama pull llama3.2:1b
```

Then edit `app.py`, change `self.gen_model`:
```python
self.gen_model = "llama3.2:1b"
```

Restart the server.

---

## License

MIT — use this however you want.
