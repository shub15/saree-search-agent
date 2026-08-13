"""
scripts/build_index.py
-----------------------
ONE-TIME script: read the product CSV, fetch images from their URLs,
embed with FashionEmbedder, and build the FAISS index.

Input:  byrappa_tejas_31july.csv (columns: Name, SKU, Stock,
        Retail Price, Discounted Price, image_url, Website Link)
Output: data/index.faiss + data/metadata.json

Usage
-----
    # Quick dry-run (first 10 rows only):
    python scripts/build_index.py --csv data/images/byrappa_tejas_31july.csv --dry-run

    # Full build:
    python scripts/build_index.py --csv data/images/byrappa_tejas_31july.csv

Notes
-----
- Images are .webp from byrappasilk.in — Pillow handles these natively.
- Fetching is parallelised with ThreadPoolExecutor for speed.
- Failed URLs are skipped with a warning (logged to build_errors.log).
- Thumbnails (200×200 JPEG, base64) are stored inside metadata.json so the
  deployed app can display results without hosting images externally.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests
from PIL import Image

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedder import FashionEmbedder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
error_logger = logging.getLogger("build_errors")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THUMB_SIZE = 200        # pixels, max dimension
THUMB_QUALITY = 72      # JPEG quality
FETCH_TIMEOUT = 20      # seconds per HTTP request
MAX_WORKERS = 8         # parallel download threads
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------
def load_csv(csv_path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalise key names (strip whitespace)
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    logger.info("Loaded %d rows from '%s'", len(rows), csv_path)
    return rows


# ---------------------------------------------------------------------------
# Image fetching
# ---------------------------------------------------------------------------
def fetch_image(url: str) -> Optional[Image.Image]:
    """Download an image from a URL and return it as a PIL RGB image."""
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers=REQUEST_HEADERS)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img
    except Exception as exc:
        logger.warning("  ✗ Failed to fetch '%s': %s", url, exc)
        return None


def make_thumbnail_b64(image: Image.Image) -> str:
    """Resize and base64-encode as JPEG data URI."""
    thumb = image.copy()
    thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=THUMB_QUALITY, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Worker: fetch + embed one row
# ---------------------------------------------------------------------------
def process_row(
    idx: int,
    row: Dict[str, str],
    embedder: FashionEmbedder,
) -> Optional[Tuple[int, np.ndarray, Dict[str, Any]]]:
    """
    Returns (original_row_index, embedding, metadata_dict) or None on failure.
    """
    url = row.get("image_url", "").strip()
    if not url:
        logger.warning("Row %d has no image_url — skipping.", idx)
        return None

    image = fetch_image(url)
    if image is None:
        return None

    try:
        embedding = embedder.embed_for_index(image)
        thumbnail_b64 = make_thumbnail_b64(image)
    except Exception as exc:
        logger.warning("Row %d embed failed: %s", idx, exc)
        return None

    meta: Dict[str, Any] = {
        "filename": f"{row.get('SKU', f'item_{idx}')}.webp",
        "name": row.get("Name", ""),
        "sku": row.get("SKU", ""),
        "retail_price": row.get("Retail Price", ""),
        "discounted_price": row.get("Discounted Price", ""),
        "stock": row.get("Stock", ""),
        "image_url": url,
        "website_link": row.get("Website Link", ""),
        "thumbnail_b64": thumbnail_b64,
    }
    return (idx, embedding, meta)


# ---------------------------------------------------------------------------
# Main build routine
# ---------------------------------------------------------------------------
def build(
    csv_path: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.faiss"
    meta_path = output_dir / "metadata.json"

    # Set up error log file
    fh = logging.FileHandler(output_dir / "build_errors.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.WARNING)
    error_logger.addHandler(fh)

    rows = load_csv(csv_path)
    if dry_run:
        rows = rows[:10]
        logger.info("DRY RUN — processing only %d rows", len(rows))

    # ── Load model ────────────────────────────────────────────────────────
    logger.info("Loading FashionEmbedder …")
    embedder = FashionEmbedder()
    logger.info("Model ready. Starting parallel fetch + embed …")

    # ── Parallel fetch + embed ────────────────────────────────────────────
    # We preserve order by keying results on the original row index.
    results: Dict[int, Tuple[np.ndarray, Dict[str, Any]]] = {}
    n_failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_row, i, row, embedder): i
            for i, row in enumerate(rows)
        }
        for done_idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is None:
                n_failed += 1
            else:
                orig_idx, emb, meta = result
                results[orig_idx] = (emb, meta)

            # Progress every 50 items
            if done_idx % 50 == 0 or done_idx == len(rows):
                elapsed = time.time() - t0
                rate = done_idx / elapsed if elapsed > 0 else 0
                eta = (len(rows) - done_idx) / rate if rate > 0 else 0
                logger.info(
                    "  %d / %d processed  (%.1f img/s, ETA %.0fs, %d failed)",
                    done_idx, len(rows), rate, eta, n_failed,
                )

    # Sort by original row order
    ordered_keys = sorted(results.keys())
    all_embeddings = [results[k][0] for k in ordered_keys]
    all_metadata   = [results[k][1] for k in ordered_keys]

    logger.info(
        "Done. %d embedded successfully, %d skipped.",
        len(all_embeddings), n_failed,
    )

    if not all_embeddings:
        logger.error("No embeddings produced — aborting.")
        sys.exit(1)

    # ── Build FAISS index ─────────────────────────────────────────────────
    matrix = np.stack(all_embeddings).astype(np.float32)
    dim = matrix.shape[1]
    logger.info("Building IndexFlatIP: %d vectors × dim %d …", len(matrix), dim)

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    # ── Save ──────────────────────────────────────────────────────────────
    faiss.write_index(index, str(index_path))
    logger.info(
        "Saved  %s  (%.2f MB)",
        index_path, index_path.stat().st_size / 1e6,
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    logger.info(
        "Saved  %s  (%.2f MB)",
        meta_path, meta_path.stat().st_size / 1e6,
    )

    # ── Sanity check ──────────────────────────────────────────────────────
    logger.info("Sanity check: querying index with first vector …")
    check_idx = faiss.read_index(str(index_path))
    scores, indices = check_idx.search(matrix[:1], 5)
    logger.info(
        "Top-5 results: indices=%s  scores=%s",
        indices[0].tolist(),
        [f"{s:.4f}" for s in scores[0].tolist()],
    )
    logger.info(
        "✓ All good!  Commit  data/index.faiss  and  data/metadata.json  to your repo."
    )
    if n_failed:
        logger.info("  %d rows failed — see  %s", n_failed, output_dir / "build_errors.log")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS index from saree product CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/images/byrappa_tejas_31july.csv"),
        help="Path to the product CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write index.faiss and metadata.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 10 rows to verify setup.",
    )
    args = parser.parse_args()
    build(
        csv_path=args.csv,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
