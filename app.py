"""
Custom Vector Database & AI Assistant
Built by Arvind Dwivedi

A custom vector database to store and retrieve text embeddings using cosine
similarity for semantic search, with an AI-powered RAG pipeline for
context-aware response generation.

Algorithms implemented from scratch:
  - HNSW  (Hierarchical Navigable Small World) — O(log N) approximate
  - KD-Tree                                     — O(log N) exact
  - Brute Force                                 — O(N·d) exact baseline
"""

from flask import Flask, request, jsonify, send_file
import numpy as np
import requests
import threading
import heapq
import time
import math
import random
import os

app = Flask(__name__)

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16  # demo vectors dimension (doc embeddings are 768D from Ollama)

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean_distance(a, b):
    """Euclidean (L2) distance between two vectors."""
    diff = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
    return float(np.sqrt(np.dot(diff, diff)))


def cosine_distance(a, b):
    """Cosine distance: 1 - cosine_similarity. Range [0, 2]."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = float(np.dot(a_arr, b_arr))
    na = float(np.linalg.norm(a_arr))
    nb = float(np.linalg.norm(b_arr))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)


def manhattan_distance(a, b):
    """Manhattan (L1) distance between two vectors."""
    return float(np.sum(np.abs(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32))))


def get_dist_fn(metric):
    """Return the distance function for a given metric name."""
    if metric == "cosine":
        return cosine_distance
    elif metric == "manhattan":
        return manhattan_distance
    return euclidean_distance


# =====================================================================
#  BRUTE FORCE — O(N·d) exact baseline
# =====================================================================

class BruteForce:
    """Linear scan K-NN search. Simple but exact."""

    def __init__(self):
        self.items = []

    def insert(self, item):
        self.items.append(item)

    def knn(self, query, k, dist_fn):
        results = [(dist_fn(query, item["emb"]), item["id"]) for item in self.items]
        results.sort()
        return results[:k]

    def remove(self, item_id):
        self.items = [item for item in self.items if item["id"] != item_id]


# =====================================================================
#  KD-TREE — O(log N) exact search with axis-aligned partitioning
# =====================================================================

class _KDNode:
    """Internal node for the KD-Tree."""
    __slots__ = ("item", "left", "right")

    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None


class KDTree:
    """
    K-Dimensional Tree for exact nearest-neighbor search.
    Cycles through dimensions at each level. Prunes subtrees using
    the axis-aligned distance bound (ball-within-hyperslab check).
    """

    def __init__(self, dims):
        self.root = None
        self.dims = dims

    def _insert(self, node, item, depth):
        if node is None:
            return _KDNode(item)
        axis = depth % self.dims
        if item["emb"][axis] < node.item["emb"][axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def insert(self, item):
        self.root = self._insert(self.root, item, 0)

    def _knn(self, node, query, k, depth, dist_fn, heap):
        """Recursive KNN with pruning. heap is a max-heap (negative dists)."""
        if node is None:
            return

        d = dist_fn(query, node.item["emb"])

        if len(heap) < k:
            heapq.heappush(heap, (-d, node.item["id"]))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.item["id"]))

        axis = depth % self.dims
        diff = query[axis] - node.item["emb"][axis]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, query, k, depth + 1, dist_fn, heap)

        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, query, k, depth + 1, dist_fn, heap)

    def knn(self, query, k, dist_fn):
        heap = []
        self._knn(self.root, query, k, 0, dist_fn, heap)
        results = [(-d, item_id) for d, item_id in heap]
        results.sort()
        return results

    def rebuild(self, items):
        self.root = None
        for item in items:
            self.insert(item)


# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSW:
    """
    Hierarchical Navigable Small World graph for approximate nearest-neighbor
    search. Same algorithm used by Pinecone, Weaviate, Chroma, and Milvus.

    Builds a multilayer graph where each layer is progressively sparser.
    Searches start at the top layer and zoom in, achieving O(log N) complexity.

    Parameters:
        M        — max connections per node per layer (default 16)
        ef_build — beam width during construction (default 200)
    """

    def __init__(self, M=16, ef_build=200):
        self.M = M
        self.M0 = 2 * M          # max connections at layer 0
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(M)
        self.graph = {}           # id -> {item, max_layer, neighbors}
        self.top_layer = -1
        self.entry_point = -1
        self.rng = random.Random(42)

    def _rand_level(self):
        """Assign a random max layer via exponential distribution."""
        return int(-math.log(self.rng.random() + 1e-15) * self.mL)

    def _search_layer(self, query, ep, ef, layer, dist_fn):
        """
        Beam search on a single layer. Returns up to `ef` nearest neighbors
        as sorted list of (distance, id) tuples.
        """
        visited = {ep}

        d0 = dist_fn(query, self.graph[ep]["item"]["emb"])
        candidates = [(d0, ep)]       # min-heap: closest candidates first
        found = [(-d0, ep)]           # max-heap (neg dist): farthest on top

        while candidates:
            cd, cid = heapq.heappop(candidates)

            if len(found) >= ef and cd > -found[0][0]:
                break

            node = self.graph.get(cid)
            if node is None or layer >= len(node["neighbors"]):
                continue

            for nid in node["neighbors"][layer]:
                if nid in visited or nid not in self.graph:
                    continue
                visited.add(nid)
                nd = dist_fn(query, self.graph[nid]["item"]["emb"])

                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(candidates, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        results = [(-d, nid) for d, nid in found]
        results.sort()
        return results

    def _select_neighbors(self, candidates, max_m):
        """Select top max_m neighbors from sorted candidates."""
        return [cid for _, cid in candidates[:max_m]]

    def insert(self, item, dist_fn):
        """Insert a vector into the HNSW graph."""
        item_id = item["id"]
        level = self._rand_level()
        self.graph[item_id] = {
            "item": item,
            "max_layer": level,
            "neighbors": [[] for _ in range(level + 1)],
        }

        if self.entry_point == -1:
            self.entry_point = item_id
            self.top_layer = level
            return

        ep = self.entry_point

        # Greedy descent from top layer down to level+1
        for lc in range(self.top_layer, level, -1):
            if lc < len(self.graph[ep]["neighbors"]):
                W = self._search_layer(item["emb"], ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        # Insert at each layer from min(top_layer, level) down to 0
        for lc in range(min(self.top_layer, level), -1, -1):
            W = self._search_layer(item["emb"], ep, self.ef_build, lc, dist_fn)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_neighbors(W, max_m)
            self.graph[item_id]["neighbors"][lc] = sel

            # Add bidirectional connections and prune if needed
            for nid in sel:
                if nid not in self.graph:
                    continue
                node = self.graph[nid]
                while len(node["neighbors"]) <= lc:
                    node["neighbors"].append([])
                conn = node["neighbors"][lc]
                conn.append(item_id)

                if len(conn) > max_m:
                    dists = [
                        (dist_fn(node["item"]["emb"], self.graph[c]["item"]["emb"]), c)
                        for c in conn if c in self.graph
                    ]
                    dists.sort()
                    node["neighbors"][lc] = [c for _, c in dists[:max_m]]

            if W:
                ep = W[0][1]

        if level > self.top_layer:
            self.top_layer = level
            self.entry_point = item_id

    def knn(self, query, k, ef, dist_fn):
        """K-nearest-neighbor search across the full HNSW graph."""
        if self.entry_point == -1:
            return []

        ep = self.entry_point
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.graph[ep]["neighbors"]):
                W = self._search_layer(query, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        W = self._search_layer(query, ep, max(ef, k), 0, dist_fn)
        return W[:k]

    def remove(self, item_id):
        """Remove a node from the graph (cleans all edges)."""
        if item_id not in self.graph:
            return
        for nid, node in self.graph.items():
            for layer in node["neighbors"]:
                if item_id in layer:
                    layer.remove(item_id)

        if self.entry_point == item_id:
            self.entry_point = -1
            for nid in self.graph:
                if nid != item_id:
                    self.entry_point = nid
                    break

        del self.graph[item_id]

    def get_info(self):
        """Return graph structure info for visualization."""
        top = self.top_layer
        count = len(self.graph)
        max_l = max(top + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes_list = []
        edges_list = []

        for nid, node in self.graph.items():
            nodes_list.append({
                "id": nid,
                "metadata": node["item"]["metadata"],
                "category": node["item"]["category"],
                "maxLyr": node["max_layer"],
            })
            for lc in range(min(node["max_layer"] + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(node["neighbors"]):
                    for nbr in node["neighbors"][lc]:
                        if nid < nbr:
                            edges_per_layer[lc] += 1
                            edges_list.append({"src": nid, "dst": nbr, "lyr": lc})

        return {
            "topLayer": top,
            "nodeCount": count,
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes_list,
            "edges": edges_list,
        }

    def size(self):
        return len(self.graph)


# =====================================================================
#  VECTOR DATABASE — unified interface over all 3 algorithms (16D demo)
# =====================================================================

class VectorDB:
    """
    Custom vector database with three search backends.
    Stores and retrieves text embeddings using cosine similarity
    for semantic search. Thread-safe.
    """

    def __init__(self, dims):
        self.dims = dims
        self.store = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.lock = threading.Lock()
        self.next_id = 1

    def insert(self, metadata, category, emb, dist_fn):
        with self.lock:
            item = {"id": self.next_id, "metadata": metadata,
                    "category": category, "emb": emb}
            self.next_id += 1
            self.store[item["id"]] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist_fn)
            return item["id"]

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, query, k, metric, algo):
        with self.lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(query, k, dist_fn)
            elif algo == "kdtree":
                raw = self.kdt.knn(query, k, dist_fn)
            else:
                raw = self.hnsw.knn(query, k, 50, dist_fn)

            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, item_id in raw:
                if item_id in self.store:
                    item = self.store[item_id]
                    hits.append({
                        "id": item_id,
                        "metadata": item["metadata"],
                        "category": item["category"],
                        "embedding": item["emb"],
                        "distance": round(d, 6),
                    })
            return {"results": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, query, k, metric):
        with self.lock:
            dist_fn = get_dist_fn(metric)

            def time_fn(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            return {
                "bruteforceUs": time_fn(lambda: self.bf.knn(query, k, dist_fn)),
                "kdtreeUs": time_fn(lambda: self.kdt.knn(query, k, dist_fn)),
                "hnswUs": time_fn(lambda: self.hnsw.knn(query, k, 50, dist_fn)),
                "itemCount": len(self.store),
            }

    def all_items(self):
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self):
        with self.lock:
            return self.hnsw.get_info()

    def size(self):
        with self.lock:
            return len(self.store)


# =====================================================================
#  OLLAMA CLIENT — wraps local Ollama REST API
# =====================================================================

class OllamaClient:
    """
    HTTP client for the local Ollama REST API.
    Handles embedding generation (nomic-embed-text) and
    text generation (llama3.2) for the RAG pipeline.
    """

    def __init__(self, host="127.0.0.1", port=11434):
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text):
        """Generate embedding vector for text using nomic-embed-text."""
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt):
        """Generate text using the LLM (llama3.2)."""
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"


# =====================================================================
#  DOCUMENT DATABASE — HNSW over real Ollama embeddings (768D)
# =====================================================================

class DocumentDB:
    """
    Document store with HNSW-based semantic search over real embeddings.
    Handles document indexing, embedding storage, and retrieval for the
    RAG pipeline. Thread-safe.
    """

    def __init__(self):
        self.store = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.lock = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title, text, emb):
        """Insert a document chunk with its pre-computed embedding."""
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            item_id = self.next_id
            self.next_id += 1
            doc = {"id": item_id, "title": title, "text": text, "emb": emb}
            self.store[item_id] = doc
            vi = {"id": item_id, "metadata": title, "category": "doc", "emb": emb}
            self.hnsw.insert(vi, cosine_distance)
            self.bf.insert(vi)
            return item_id

    def search(self, query, k, max_dist=0.7):
        """Semantic search — returns top-k most similar document chunks."""
        with self.lock:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(query, k, cosine_distance)
            else:
                raw = self.hnsw.knn(query, k, 50, cosine_distance)
            results = []
            for d, item_id in raw:
                if item_id in self.store and d <= max_dist:
                    results.append((d, self.store[item_id]))
            return results

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.hnsw.remove(item_id)
            self.bf.remove(item_id)
            return True

    def all_docs(self):
        with self.lock:
            return list(self.store.values())

    def size(self):
        with self.lock:
            return len(self.store)

    def get_dims(self):
        return self.dims


# =====================================================================
#  TEXT CHUNKER — overlapping window splitter
# =====================================================================

def chunk_text(text, chunk_words=250, overlap_words=30):
    """
    Split text into overlapping chunks for embedding.
    Each chunk is ~250 words with 30-word overlap for context continuity.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks


# =====================================================================
#  DEMO DATA — 20 pre-loaded 16D semantic vectors across 4 categories
# =====================================================================

def load_demo(db):
    """Load 20 demo vectors: CS, Math, Food, Sports (dims 0-3, 4-7, 8-11, 12-15)."""
    dist_fn = get_dist_fn("cosine")
    demos = [
        # ── Computer Science ──
        ("Linked List: nodes connected by pointers", "cs",
         [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        # ── Mathematics ──
        ("Calculus: derivatives integrals and limits", "math",
         [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        # ── Food ──
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        # ── Sports ──
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for meta, cat, emb in demos:
        db.insert(meta, cat, emb, dist_fn)


# =====================================================================
#  INITIALIZE DATABASE & OLLAMA
# =====================================================================

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)

ollama_up = ollama.is_available()
print("\n" + "=" * 55)
print("  Custom Vector Database & AI Assistant")
print("  Built by Arvind Dwivedi")
print("=" * 55)
print(f"  Server:     http://localhost:8080")
print(f"  Demo:       {db.size()} vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
print(f"  Ollama:     {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
if ollama_up:
    print(f"  Embed:      {ollama.embed_model}")
    print(f"  Generate:   {ollama.gen_model}")
print("=" * 55 + "\n")


# =====================================================================
#  REST API — Demo Vector Endpoints
# =====================================================================

@app.route("/search")
def api_search():
    v = request.args.get("v", "")
    query = []
    for x in v.split(","):
        try:
            query.append(float(x))
        except ValueError:
            pass
    if len(query) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})

    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    algo = request.args.get("algo", "hnsw")

    return jsonify(db.search(query, k, metric, algo))


@app.route("/insert", methods=["POST"])
def api_insert():
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid body"})
    meta = data.get("metadata", "")
    cat = data.get("category", "")
    emb = data.get("embedding", [])
    if not meta or not emb or len(emb) != DIMS:
        return jsonify({"error": "invalid body"})

    item_id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": item_id})


@app.route("/delete/<int:item_id>", methods=["DELETE"])
def api_delete(item_id):
    return jsonify({"ok": db.remove(item_id)})


@app.route("/items")
def api_items():
    return jsonify([
        {"id": v["id"], "metadata": v["metadata"],
         "category": v["category"], "embedding": v["emb"]}
        for v in db.all_items()
    ])


@app.route("/benchmark")
def api_benchmark():
    v = request.args.get("v", "")
    query = []
    for x in v.split(","):
        try:
            query.append(float(x))
        except ValueError:
            pass
    if len(query) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})

    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    return jsonify(db.benchmark(query, k, metric))


@app.route("/hnsw-info")
def api_hnsw_info():
    return jsonify(db.hnsw_info())


@app.route("/stats")
def api_stats():
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    })


# =====================================================================
#  REST API — Document + RAG Endpoints
# =====================================================================

@app.route("/doc/insert", methods=["POST"])
def api_doc_insert():
    """Chunk text, embed each chunk via Ollama, store in DocumentDB."""
    data = request.get_json()
    title = data.get("title", "")
    text = data.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"})

    chunks = chunk_text(text, 250, 30)
    ids = []

    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com "
                         "then run: ollama pull nomic-embed-text && ollama pull llama3.2"
            })
        chunk_title = (f"{title} [{i+1}/{len(chunks)}]"
                       if len(chunks) > 1 else title)
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.route("/doc/delete/<int:item_id>", methods=["DELETE"])
def api_doc_delete(item_id):
    return jsonify({"ok": doc_db.remove(item_id)})


@app.route("/doc/list")
def api_doc_list():
    docs = doc_db.all_docs()
    return jsonify([{
        "id": d["id"],
        "title": d["title"],
        "preview": d["text"][:120] + ("…" if len(d["text"]) > 120 else ""),
        "words": len(d["text"].split()),
    } for d in docs])


@app.route("/doc/search", methods=["POST"])
def api_doc_search():
    """Fast retrieval for the UI visualizer."""
    data = request.get_json()
    question = data.get("question", "")
    k = data.get("k", 3)
    if not question:
        return jsonify({"error": "need question"})

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    hits = doc_db.search(q_emb, k)
    return jsonify({
        "contexts": [
            {"id": doc["id"], "title": doc["title"], "distance": round(d, 4)}
            for d, doc in hits
        ]
    })


@app.route("/doc/ask", methods=["POST"])
def api_doc_ask():
    """
    Full RAG pipeline: embed question → retrieve context → generate answer.
    This is the core of the AI-powered Retrieval-Augmented Generation system.
    """
    data = request.get_json()
    question = data.get("question", "")
    k = data.get("k", 3)
    if not question:
        return jsonify({"error": "need question"})

    # Step 1: Embed the question
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    # Step 2: Retrieve top-k relevant chunks via HNSW semantic search
    hits = doc_db.search(q_emb, k)

    # Step 3: Build context-aware prompt
    ctx_text = ""
    for i, (d, doc) in enumerate(hits):
        ctx_text += f"[{i+1}] {doc['title']}:\n{doc['text']}\n\n"

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say "
        "things like 'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        f"Context:\n{ctx_text}"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Step 4: Generate answer via local LLM
    answer = ollama.generate(prompt)

    # Step 5: Return answer with retrieved context metadata
    return jsonify({
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [
            {"id": doc["id"], "title": doc["title"],
             "text": doc["text"], "distance": round(d, 4)}
            for d, doc in hits
        ],
        "docCount": doc_db.size(),
    })


@app.route("/status")
def api_status():
    up = ollama.is_available()
    return jsonify({
        "ollamaAvailable": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size(),
    })


# ── Serve the frontend ────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_file("index.html")


# ── CORS ──────────────────────────────────────────────────────────

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
