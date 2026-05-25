# 🎵 RAG-Based Content Recommender — Amazon Digital Music

A complete **Retrieval-Augmented Generation (RAG)** pipeline for content-based music recommendation, built on top of the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) Digital Music dataset. The system combines semantic vector retrieval (FAISS) with a large language model (Gemini) to deliver personalized, explainable music recommendations.

> Developed as part of the *Information Retrieval and Recommendation Systems* course  
> Master's in Data Science and Computer Engineering

---

## 📐 Architecture

The system follows the canonical RAG architecture across six modules:

```
Documents → Chunking → Embedding Model → Vector Store (FAISS)
                                              ↓
User Profile (Query) → Embedding → Search → Retrieve top-K
                                              ↓
                              Augment (Profile + Candidates) → Prompt
                                              ↓
                                        LLM (Gemini) → Re-ranked JSON
                                              ↓
                                   Hallucination Filter → Response
```

| Module | Description |
|--------|-------------|
| **I — Item Processing** | HTML cleaning, lowercasing, metadata + review concatenation |
| **II — Indexing & Retrieval** | `all-MiniLM-L6-v2` embeddings, L2 norm, FAISS `IndexFlatIP` |
| **III — Augmentation** | Structured prompt: role + user profile + candidates + instructions |
| **IV — Generation** | Gemini 1.5 Flash re-ranks top-10 candidates into a structured JSON |
| **V — Hallucination Filter** | ASIN + title double-check against the input candidate list |
| **VI — Evaluation** | Leave-one-out IR metrics: P@K, R@K, NDCG@K, MRR, Hit Rate |

---

## 📂 Repository Structure

```
├── RAG_Amazon.ipynb          # Main Google Colab notebook (full pipeline)
├── report/
│   └── main.tex              # LaTeX report (Spanish)
├── data/                     # Place dataset files here (not tracked by git)
│   ├── Digital_Music.jsonl
│   └── meta_Digital_Music.jsonl
└── README.md
```

---

## 📦 Dataset

**Amazon Reviews 2023 — Digital Music** subcategory.

| File | Records | Key fields |
|------|---------|------------|
| `Digital_Music.jsonl` | 130,434 | `rating`, `title`, `text`, `user_id`, `parent_asin` |
| `meta_Digital_Music.jsonl` | 70,537 | `title`, `store`, `categories`, `features`, `description`, `details` |

Download from the official source: [https://amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/main.html)

The join key between both files is `parent_asin`.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-music-recommender.git
cd rag-music-recommender
```

### 2. Install dependencies

```bash
pip install sentence-transformers faiss-cpu tqdm beautifulsoup4 google-generativeai
```

### 3. Set up your Gemini API key

Get a free key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and set it in the notebook:

```python
GEMINI_API_KEY = "your-api-key-here"
```

### 4. Place the dataset files

Copy `Digital_Music.jsonl` and `meta_Digital_Music.jsonl` into the `data/` folder (or update the paths in the notebook).

### 5. Run the notebook

Open `RAG_Amazon.ipynb` in Google Colab (GPU runtime recommended) and execute all cells in order.

---

## ⚙️ Key Design Decisions

**Item document construction**: each item's document combines metadata fields (`title`, `store/artist`, `categories`, `features`, `description`, `details`, `average_rating`) with up to 5 aggregated user reviews, separated by `|`. This enrichment is critical because `description` and `categories` fields are empty in a large fraction of items.

**Embedding model**: `all-MiniLM-L6-v2` (384 dimensions) was chosen for its balance between semantic quality and computational efficiency. All vectors are L2-normalized so that the inner product computed by FAISS `IndexFlatIP` is equivalent to cosine similarity.

**User profiles**: 6 profiles were built using two strategies — 3 real users (identified from top reviewers) whose profile is the concatenation of their highest-rated item documents, and 3 synthetic profiles with contrasting musical preferences (classical, hip-hop, jazz).

**Prompt design**: the LLM prompt enforces four strict instructions — RE-RANKING, JUSTIFICATION, QUALITY FILTER, and INTEGRITY — with the last one explicitly forbidding the model from suggesting items not present in the input candidate list.

**Hallucination filter**: post-generation check verifying that every item returned by the LLM exists in the input candidate list, validated by both ASIN and normalized title.

---

## 📊 Results

### RAG Generation (6 user profiles)

| Profile | #1 Recommendation | Hallucinations | Re-ranks | Valid items |
|---------|-------------------|---------------|----------|-------------|
| Real #1 (Contemporary classical) | Laborintus II | 0 | 1 | 3 |
| Real #2 (Beatles collector) | Not a Second Time | 0 | 2 | 3 |
| Real #3 (Piano classics) | Debussy: Das Gesamtwerk | 0 | 2 | 3 |
| Synthetic — Classical | The Classical Music for Reading | 0 | 3 | 3 |
| Synthetic — Hip-Hop | Control System (Ab-Soul) | 0 | 2 | 3 |
| Synthetic — Jazz | 17 Classic Albums (Miles Davis) | 0 | 2 | 3 |
| **Total** | | **0 (0%)** | **12 (66.7%)** | **18** |

### Retrieval Evaluation (300 users, leave-one-out)

| Metric | K=5 | K=10 | K=20 |
|--------|-----|------|------|
| Precision@K | 0.0127 | 0.0083 | 0.0048 |
| Recall@K | 0.0633 | 0.0833 | 0.0967 |
| NDCG@K | 0.0330 | 0.0394 | 0.0426 |
| MRR | — | 0.0263 | — |
| Hit Rate | — | — | 0.0967 |

> **Note on metric values**: low absolute values are expected in this evaluation scenario. There is a single relevant item per user in a search space of 70,537 candidates. A Hit Rate@20 of ~9.7% means the system places the relevant item in the top 20 for almost 1 in 10 users — a solid result for purely content-based retrieval with no collaborative signal. When the system does find the relevant item, it places it at an average position of **6.1** within the ranking.

---

## 🔬 Qualitative Highlights

The LLM re-ranking adds measurable value beyond pure vector similarity. A notable example is the **jazz synthetic profile**: FAISS ranked *Portrait in Jazz* (Bill Evans) at position 8, but Gemini promoted it to position 3, explicitly reasoning that Bill Evans is a favorite artist mentioned in the profile and that the album showcases the complex harmonies and improvisation the user seeks. This level of fine-grained semantic reasoning over profile content is not achievable through embedding similarity alone.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector store | `faiss-cpu` — `IndexFlatIP` |
| LLM | Google Gemini 1.5 Flash |
| Text cleaning | `beautifulsoup4`, `re` |
| Data handling | `pandas`, `numpy` |
| Environment | Google Colab (GPU T4) |

---

## 📈 Possible Improvements

- Use a domain-specific embedding model (e.g., fine-tuned on music metadata)
- Incorporate collaborative signals to enrich user profiles
- Explore approximate FAISS indexes (`IndexIVFFlat`) for larger catalogs
- Apply chunking strategies for items with very long descriptions
- Implement a multi-stage retrieval pipeline (BM25 + dense retrieval)

---

## 📄 License

This project is intended for academic purposes. The Amazon Reviews 2023 dataset is subject to its own terms of use — please refer to the [official dataset page](https://amazon-reviews-2023.github.io/main.html).

---

## 📚 References

1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
2. Zhao, P., et al. (2025). *RAG for AI-Generated Content: A Survey*. Data Science and Engineering.
3. Hou, Y., et al. (2024). *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952.
4. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT*. EMNLP 2019.
5. Johnson, J., Douze, M. & Jégou, H. (2021). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data.
