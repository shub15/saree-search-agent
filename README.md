# 🪡 TailorTalk — AI Saree Similarity Search

A conversational AI agent that finds visually similar sarees from a fashion catalogue.
Upload any saree image and the agent retrieves the closest visual matches by colour,
fabric texture, weave pattern, border, and pallu work.

**Live Demo**: _[link after deploy]_  
**GitHub**: _[this repo]_

---

## ✨ Features

- 🔍 **Fine-grained image similarity** — distinguishes subtle differences in fabric and print
- 💬 **Natural chat interface** — Gemini 1.5 Flash orchestrates tool calls conversationally
- 🖼️ **Image upload or URL** — works with local files or any public image link
- ⚡ **Instant search** — pre-built FAISS index, no re-embedding at query time
- 🎨 **Premium dark UI** — glassmorphism design with animated result cards

---

## 🏗️ Architecture

```
User (image + chat)
       │
       ▼
 Streamlit App (app.py)
       │ LangChain AgentExecutor
       ▼
 Gemini 1.5 Flash ──calls──▶  find_similar_sarees tool (agent.py)
                                        │
                                        ▼
                          FashionEmbedder.embed_query()  (embedder.py)
                          ├── Marqo-FashionSigLIP visual features
                          └── HSV colour histogram (fused 85/15)
                                        │
                                        ▼
                          FAISS IndexFlatIP cosine search  (search.py)
                                        │
                                        ▼
                          Top-K results (filename + score + thumbnail)
```

---

## 🧠 Model & Technology Choices

### Embedding Model: `Marqo/marqo-fashionSigLIP`

| Property | Detail |
|---|---|
| Base | SigLIP ViT-L/14 |
| Training | Generalised Contrastive Learning (GCL) on 7 fashion aspects |
| Fashion aspects | color, material, category, style, details, title, keywords |
| License | Apache 2.0 |
| Why better than CLIP | Up to 57% better MRR@10 on fashion retrieval benchmarks |

Loaded via `open_clip` from the HuggingFace Hub (`hf-hub:Marqo/marqo-fashionSigLIP`).

### Vector Database: FAISS `IndexFlatIP`

- **Exact cosine similarity** search on L2-normalised vectors — zero quantisation loss
- Dataset size (1 075 images) easily fits in CPU RAM
- Pre-built index committed to the repo — zero startup cost in deployed app
- No external service, no API key, no running infra

### Agent Framework: LangChain + Gemini 1.5 Flash

- `@tool` decorator defines `find_similar_sarees` with a clear input/output schema
- `create_tool_calling_agent` + `AgentExecutor` handles the ReAct loop
- Shared `results_store` list (closure pattern) passes structured results back to the UI

---

## 🔬 Search Quality Improvements

This is where the results go from "generic" to "fine-grained":

### 1. Fashion-Specific Embeddings (vs. generic CLIP)
Marqo-FashionSigLIP was trained with **Generalised Contrastive Learning** specifically
to separate items by material, details, and colour — the exact attributes that differ
between sarees in the same catalogue.

### 2. Fused Colour Histogram (HSV)
A 96-dimensional HSV colour histogram is fused with the visual embedding
(weighted 85% visual + 15% colour, both L2-normalised before concat).

This provides an explicit, hard colour signal. Plain CLIP embeddings can treat
a red Banarasi and a blue Banarasi as highly similar because the weave pattern
dominates. The fused embedding corrects this.

### 3. Inner-Product Search on L2-Normalised Vectors
Using `IndexFlatIP` on L2-normalised vectors is mathematically equivalent to
exact cosine similarity — no approximation error for this dataset size.

### 4. Query-Time Multi-Crop Averaging
At query time, the embedding is averaged over two crops of the query image
(original + tight centre crop). This reduces sensitivity to background regions
and image borders, stabilising retrieval on real-world photos.

### 5. Score Calibration
Raw inner-product scores ∈ [−1, 1] are mapped to [0, 1] via
`sim = (score + 1) / 2` so user-visible scores are intuitive.

---

## 🚀 Setup & Local Development

### Prerequisites
- Python 3.10+
- [Google AI Studio API key](https://aistudio.google.com) (free)
- Dataset downloaded and unzipped

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the product CSV

The dataset is a CSV file with columns:
`Name, SKU, Stock, Retail Price, Discounted Price, image_url, Website Link`

Place it at:
```
data/images/byrappa_tejas_31july.csv
```

### 3. Build the FAISS index (one-time)

The script fetches all 1 075 product images directly from their live URLs,
embeds them, and saves the index with rich product metadata (name, price, link).

```bash
# Quick dry-run first (10 rows only):
python scripts/build_index.py --csv data/images/byrappa_tejas_31july.csv --dry-run

# Full build (~10-15 min on CPU, parallelised with 8 download threads):
python scripts/build_index.py --csv data/images/byrappa_tejas_31july.csv
```

This generates:
- `data/index.faiss` (~3 MB) — FAISS vector index
- `data/metadata.json` (~8 MB) — product info + base64 thumbnails

**Commit both files to your repo** — the deployed app reads them at startup.
Failed rows are logged to `data/build_errors.log`.

### 4. Configure your API key

```bash
# Create secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit and set your key:
# GOOGLE_API_KEY = "AIza..."
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push your repository to GitHub (including `data/index.faiss` and `data/metadata.json`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, file `app.py`
4. Click **Advanced settings** → add secret:
   ```
   GOOGLE_API_KEY = "AIza..."
   ```
5. Click **Deploy** — done ✓

> **Note**: The `data/images/` directory is gitignored (too large). The deployed app
> only needs `data/index.faiss` and `data/metadata.json` — thumbnails are stored
> as base64 inside the JSON, so no image hosting is required.

---

## 📁 Project Structure

```
tailortalk-saree-search/
├── app.py                     # Streamlit app entry point
├── agent.py                   # LangChain tool definition + Gemini agent
├── embedder.py                # Marqo-FashionSigLIP + HSV colour fusion
├── search.py                  # FAISS index wrapper + SearchResult dataclass
├── scripts/
│   └── build_index.py         # One-time dataset indexing script
├── data/
│   ├── index.faiss            # Pre-built FAISS index ✓ commit
│   ├── metadata.json          # Filenames + base64 thumbnails ✓ commit
│   └── images/               # Raw images ✗ gitignored
├── .streamlit/
│   ├── config.toml            # Dark theme configuration
│   └── secrets.toml.example   # Template (never commit the real file)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚖️ Assumptions & Trade-offs

| Decision | Trade-off |
|---|---|
| FAISS over Pinecone/Qdrant | No managed service needed; exact search suffices for 1 K images |
| Base64 thumbnails in JSON | Avoids image hosting; ~5-8 MB JSON vs. hosting 1 K images externally |
| CPU inference | Streamlit free tier has no GPU; model still runs in ~1-2s per query |
| 85/15 visual/colour split | Tuned heuristically; richer fusion (e.g. per-region attention) would improve further |
| Gemini 1.5 Flash | Free tier; GPT-4o / Claude Sonnet would give better conversational quality |
| No fine-tuning | Fine-tuning on labelled saree pairs (e.g. via triplet loss) would push quality higher |

---

## 📄 License

MIT
