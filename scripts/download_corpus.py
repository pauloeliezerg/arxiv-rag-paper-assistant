#!/usr/bin/env python3
"""
Download papers from arXiv by ID.
Usage: python scripts/download_corpus.py
"""

import os
import time
import urllib.request
from pathlib import Path

# Lista dos papers canônicos de RAG
ARXIV_IDS = [
    "2005.11401",  # RAG original (Lewis 2020)
    "2004.04906",  # DPR — Dense Passage Retrieval
    "2004.10373",  # REALM — pré-treino com retrieval
    "2112.09118",  # RETRO
    "2208.09257",  # Atlas
    "2212.10560",  # HyDE
    "2301.12652",  # REPLUG
    "2310.11511",  # Self-RAG
    "2312.10997",  # Survey RAG (Gao 2023)
    "2401.08406",  # RAG vs Fine-tuning
    "2404.16130",  # GraphRAG
    "2309.01431",  # RGB Benchmark
]

def download_paper(arxiv_id: str, output_dir: Path) -> bool:
    """Download a single paper PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    output_path = output_dir / f"{arxiv_id}.pdf"
    
    # Avoid re-downloading
    if output_path.exists():
        print(f"⏭️  Skipping {arxiv_id} (already exists)")
        return True
    
    try:
        print(f"⬇️  Downloading {arxiv_id}...")
        # Add a User-Agent header to be a good citizen
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "ArXivPaperAssistant/1.0 (Educational Project)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
        
        with open(output_path, "wb") as f:
            f.write(content)
        
        print(f"✅ Saved {arxiv_id}.pdf")
        time.sleep(1)  # Respect arXiv's rate limit (1 request per second)
        return True
    except Exception as e:
        print(f"❌ Failed to download {arxiv_id}: {e}")
        return False

def main():
    # Define o caminho para data/corpus/ a partir da raiz do projeto
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    corpus_dir = project_root / "data" / "corpus"
    
    # Cria o diretório se ele não existir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Saving papers to: {corpus_dir}")
    
    success_count = 0
    for arxiv_id in ARXIV_IDS:
        if download_paper(arxiv_id, corpus_dir):
            success_count += 1
    
    print(f"\n📊 Download complete: {success_count}/{len(ARXIV_IDS)} papers downloaded.")

if __name__ == "__main__":
    main()
    