"""
Utilities for building a training dataset from Project Gutenberg texts.

This script:
1) Downloads the Gutenberg RDF catalog.
2) Extracts English book IDs.
3) Downloads raw book text for each ID.
4) Cleans, sentence-splits, and chunks the text.
5) Appends chunks to a newline-delimited dataset file.
"""

import requests
import time
import gzip
import xml.etree.ElementTree as ET
import re
from io import BytesIO
import tarfile

OUTPUT_FILE = "stories_dataset.txt"

# -----------------------------
# CLEANING + CHUNKING (IMPORTANT)
# -----------------------------
def clean_text(text):
    """Normalize line endings and collapse extra whitespace."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text):
    """Split text into sentences using basic punctuation boundaries."""
    return re.split(r"(?<=[.!?]) +", text)


def chunk_sentences(sentences, max_words=120):
    """Group sentences into chunks capped by a word count budget."""
    chunks = []
    current = []
    count = 0

    for s in sentences:
        words = s.split()
        if count + len(words) > max_words:
            # Close the current chunk and reset the counter.
            chunks.append(" ".join(current))
            current = []
            count = 0
        current.append(s)
        count += len(words)

    if current:
        # Append the final partial chunk.
        chunks.append(" ".join(current))

    return chunks


def append_chunks(text):
    """Clean text, chunk it, and append valid chunks to OUTPUT_FILE."""
    text = clean_text(text)
    sentences = split_sentences(text)
    chunks = chunk_sentences(sentences)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for c in chunks:
            # Skip very short chunks to reduce low-signal samples.
            if len(c) > 50:
                f.write(c + "\n")


# -----------------------------
# DOWNLOAD GUTENBERG INDEX
# -----------------------------
def download_catalog():
    """Download the Gutenberg RDF catalog to a local tarball."""
    print("[INFO] Downloading Gutenberg catalog...")
    url = "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"
    res = requests.get(url, stream=True)

    with open("catalog.tar.bz2", "wb") as f:
        # Stream to disk to avoid holding the file in memory.
        for chunk in res.iter_content(1024):
            f.write(chunk)

    print("[INFO] Catalog downloaded.")


# -----------------------------
# PARSE RDF FILES
# -----------------------------
def extract_book_ids(max_books=500):
    """Parse RDF files and return a list of English-language book IDs."""
    print("[INFO] Extracting book IDs...")

    book_ids = []

    with tarfile.open("catalog.tar.bz2", "r:bz2") as tar:
        members = tar.getmembers()

        for member in members:
            if not member.name.endswith(".rdf"):
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            try:
                tree = ET.parse(f)
                root = tree.getroot()

                # Look for English language tag.
                langs = root.findall(".//{http://purl.org/dc/terms/}language")
                if not any("en" in ET.tostring(l).decode() for l in langs):
                    continue

                # Extract the Gutenberg numeric ID from the ebook node.
                ebook = root.find(".//{http://www.gutenberg.org/2009/pgterms/}ebook")
                if ebook is not None:
                    book_id = ebook.attrib.get(
                        "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
                    )
                    if book_id:
                        book_id = book_id.split("/")[-1]
                        book_ids.append(book_id)

            except Exception:
                # Skip malformed or unexpected RDF entries.
                continue

            if len(book_ids) >= max_books:
                break

    print(f"[INFO] Collected {len(book_ids)} book IDs")
    return book_ids


# -----------------------------
# DOWNLOAD BOOK TEXT
# -----------------------------
def fetch_book(book_id):
    """Try common Gutenberg URLs for a book and return raw text or None."""
    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    for url in urls:
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                return res.text
        except Exception:
            # Try the next URL on any request failure.
            continue

    return None


# -----------------------------
# REMOVE HEADER/FOOTER
# -----------------------------
def strip_gutenberg(text):
    """Remove standard Gutenberg header/footer markers if present."""
    start = text.find("*** START")
    end = text.find("*** END")
    if start != -1 and end != -1:
        return text[start:end]
    return text


# -----------------------------
# MAIN SCRAPER
# -----------------------------
def scrape_books(max_books=500):
    """Orchestrate catalog parsing, book download, and dataset writing."""
    book_ids = extract_book_ids(max_books)

    print("[INFO] Starting downloads...")

    for i, book_id in enumerate(book_ids):
        print(f"[{i+1}/{len(book_ids)}] Book {book_id}")

        text = fetch_book(book_id)
        if not text:
            continue

        text = strip_gutenberg(text)
        append_chunks(text)

        # Be polite to Project Gutenberg's servers.
        time.sleep(0.5)

    print("[INFO] Done scraping books.")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    # Step 1: Download catalog (only once).
    download_catalog()

    # Step 2: Scrape books.
    scrape_books(max_books=2000)  # increase to 1000+ later
