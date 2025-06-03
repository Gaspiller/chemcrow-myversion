import os
import json
import requests
from urllib.parse import quote

def search_papers(query, pdir="query", semantic_scholar_api_key=None):
    """
    Search papers using Semantic Scholar API and download PDFs if available.
    Returns a dictionary: {filepath: metadata}
    """
    if not os.path.exists(pdir):
        os.makedirs(pdir)

    headers = {}
    if semantic_scholar_api_key:
        headers["x-api-key"] = semantic_scholar_api_key

    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote(query)}&limit=5&fields=title,url,externalIds,openAccessPdf"

    response = requests.get(url, headers=headers)
    try:
        papers = response.json()["data"]
    except Exception:
        print("[!] Failed to fetch paper list")
        return {}

    result = {}
    for paper in papers:
        title = paper.get("title", "paper").replace("/", "_")
        citation = paper.get("url", "")
        pdf_url = paper.get("openAccessPdf", {}).get("url", "")

        if not pdf_url:
            continue

        fname = f"{pdir}/{title}.pdf"
        try:
            r = requests.get(pdf_url, timeout=10)
            with open(fname, "wb") as f:
                f.write(r.content)
            result[fname] = {"citation": citation}
        except Exception:
            continue

    return result
