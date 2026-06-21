#!/usr/bin/env python3
"""
Debug script to understand what's happening with article extraction.
"""
import sys
sys.path.insert(0, '.')

from module.pdf_to_json.pipeline import pdf_to_json
from pathlib import Path

def debug_one_doc(pdf_path):
    """Run translation and print detailed info about what was extracted."""
    print(f"\n{'='*80}")
    print(f"DEBUG: Processing {Path(pdf_path).name}")
    print(f"{'='*80}")

    result = pdf_to_json(str(pdf_path))

    # Print metadata
    meta = result.get("magazine", {})
    editor = meta.get("editor", "MISSING")
    issue_date = meta.get("issue", {}).get("date", "MISSING")
    volume = meta.get("issue", {}).get("volume", 0)
    number = meta.get("issue", {}).get("number", 0)

    print(f"EDITOR: {editor}")
    print(f"ISSUE DATE: {issue_date}")
    print(f"VOLUME: {volume}, NUMBER: {number}")

    articles = result.get("articles", [])
    print(f"\nARTICLE COUNT: {len(articles)}")

    if len(articles) > 0:
        for i, art in enumerate(articles[:10]):  # Show first 10
            print(f"\nArticle {i}:")
            print(f"  Title: '{art.get('title', 'NO TITLE')}'")
            print(f"  Kind: {art.get('kind', 'N/A')}")
            text = art.get('text', [])
            pages = art.get('pages', [])
            print(f"  Text chunks: {len(text)}")
            if text:
                for j, chunk in enumerate(text[:3]):  # Show first 3 chunks
                    preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    print(f"    Chunk {j}: '{preview}'")
                if len(text) > 3:
                    print(f"    ... and {len(text) - 3} more")
            print(f"  Pages: {pages}")

        if len(articles) > 10:
            print(f"\n... AND {len(articles) - 10} MORE ARTICLES")

if __name__ == "__main__":
    # Debug with a small sample of training docs
    train_dir = Path("data/train")
    if not train_dir.exists():
        print("ERROR: data/train/ directory not found")
        sys.exit(1)

    doc_dirs = sorted(train_dir.glob("*"))

    # Test with just a few docs to keep it fast
    test_docs = [d for d in doc_dirs if d.name in ["cycling-vol51-2630", "the-bearings-vol5-18"]]

    for doc_dir in test_docs:
        pdf_path = doc_dir / "original.pdf"
        if pdf_path.exists():
            debug_one_doc(pdf_path)
        else:
            print(f"WARNING: {pdf_path} not found")