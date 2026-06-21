#!/usr/bin/env python3
"""
Simple debug to see what's happening with article extraction.
"""
import sys
sys.path.insert(0, '.')

# Import just what we need
from module.pdf_to_json.consolidate import consolidate, ArticleIR

def test_consolidate():
    """Test consolidate function with sample data."""
    # Create some test articles like what might come from vision model
    test_articles = [
        {
            "title": "MORE ABOUT GEARCASES",
            "kind": "prose",
            "text_chunks": [],
            "pages": [1],
            "page_index": 1
        },
        {
            "title": "Racing Techniques",
            "kind": "prose",
            "text_chunks": ["This is a paragraph about racing techniques."],
            "pages": [2],
            "page_index": 2
        },
        {
            "title": "Test Article",
            "kind": "verse",
            "text_chunks": ["Line 1", "Line 2", "Line 3"],
            "pages": [3],
            "page_index": 3
        }
    ]

    print("Testing consolidate with sample articles...")
    result = consolidate(test_articles, "dummy.pdf")

    print(f"Consolidate returned {len(result)} ArticleIR objects")
    for i, art in enumerate(result):
        print(f"  Article {i}:")
        print(f"    Title: '{art.title}'")
        print(f"    Kind: {art.kind}")
        print(f"    Text chunks: {len(art.text)}")
        for j, chunk in enumerate(art.text[:2]):
            print(f"      Chunk {j}: '{chunk[:50]}...'")

if __name__ == "__main__":
    test_consolidate()