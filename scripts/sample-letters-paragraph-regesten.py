"""
Sample letters that have regest annotations, chunk each
letter by its regest paragraphs (<div corresp="regestN">), and produce a CSV
with THREE text variants per chunk:

  - text            : plain letter text only 
  - regest_text     : the scholarly summary for a paragraph
  - text_with_regest: regest_text + letter text combined

Only letters that actually have regest-tagged divs are included -- letters
without regest structure are skipped entirely

"""

import os
import random
from pathlib import Path

import pandas as pd
from lxml import etree
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────
LETTERS_DIR = Path("../bullinger-korpus-tei/data/letters") # BULLINGER LETTERS PATH XML-TEI
OUTPUT_CSV  = Path("../data/lloom_regest_comparison_sample.csv")

N_SAMPLES = 50          # number of LETTERS to sample, among those with regesten
SEED      = 42
GLOB_PATTERN = "*.xml"

# ── TEI namespace / tag constants ───────────────────────────────────────────
NS = {"tei": "http://www.tei-c.org/ns/1.0"}
TEI_NS = "{http://www.tei-c.org/ns/1.0}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

SKIP_TAGS = {
    f"{TEI_NS}note",
    f"{TEI_NS}persName",
    f"{TEI_NS}placeName",
}
LB_TAG = f"{TEI_NS}lb"
P_TAG  = f"{TEI_NS}p"


# ── Text extraction helpers ─────────────────────────────────────────────────
def get_sentence_text(element, is_root=True):
    if element.tag in SKIP_TAGS:
        return element.tail.strip() if element.tail and element.tail.strip() else ""
    parts = []
    if element.text and element.text.strip():
        parts.append(element.text.strip())
    for child in element:
        child_text = get_sentence_text(child, is_root=False)
        if child_text:
            parts.append(child_text)
    if not is_root and element.tail and element.tail.strip():
        parts.append(element.tail.strip())
    return " ".join(parts)


def walk_body_lb(elem, parts):
    if elem.tag in SKIP_TAGS:
        if elem.tail and elem.tail.strip():
            parts.append(elem.tail.strip())
        return
    if elem.tag == LB_TAG:
        if elem.tail and elem.tail.strip():
            parts.append(elem.tail.strip())
    if elem.tag == P_TAG:
        if elem.text and elem.text.strip():
            parts.append(elem.text.strip())
    for child in elem:
        walk_body_lb(child, parts)


def extract_div_text(div):
    """Extract language-split text from a single <div>."""
    latin_sentences, enhg_sentences, rest_sentences = [], [], []

    s_tags = div.findall(".//tei:s", NS)
    if s_tags:
        for s in s_tags:
            lang = s.get(XML_LANG, "")
            text = get_sentence_text(s, is_root=True).strip()
            if not text:
                continue
            if lang == "la":
                latin_sentences.append(text)
            elif lang == "de":
                enhg_sentences.append(text)
            else:
                rest_sentences.append(text)
    else:
        # Fallback: no <s> tags in this div, walk <lb>/<p> directly
        full_text_parts = []
        walk_body_lb(div, full_text_parts)
        full_text = " ".join(full_text_parts)
        if full_text:
            rest_sentences.append(full_text)

    return latin_sentences, enhg_sentences, rest_sentences


def get_regest_lookup(root):
    """Map regest paragraph xml:id (e.g. 'regest1') -> summary text."""
    lookup = {}
    for p in root.findall(".//tei:msContents//tei:summary//tei:p", NS):
        regest_id = p.get(XML_ID, "")
        if not regest_id:
            continue
        text = get_sentence_text(p, is_root=True).strip()
        lookup[regest_id] = text
    return lookup


def has_regest_structure(root):
    """Check whether this letter has both a summary and regest-tagged divs."""
    regest_lookup = get_regest_lookup(root)
    if not regest_lookup:
        return False
    divs_with_corresp = root.findall(".//tei:body/tei:div[@corresp]", NS)
    return len(divs_with_corresp) > 0


def extract_letter_chunks_with_regest(filepath):
    """Return a list of chunk dicts, one per regest-tagged div, including
    both plain text and regest-augmented text. Returns [] if the letter
    has no regest structure."""
    try:
        tree = etree.parse(str(filepath))
        root = tree.getroot()
    except Exception as e:
        print(f"Parse error: {filepath}: {e}")
        return []

    source = root.get("source", "")
    if source == "keine":
        return []

    if not has_regest_structure(root):
        return []

    letter_id = root.get(XML_ID, os.path.splitext(os.path.basename(filepath))[0])

    sent = root.find(".//tei:correspAction[@type='sent']/tei:date", NS)
    date = sent.get("when") if sent is not None else None

    regest_lookup = get_regest_lookup(root)
    divs_with_corresp = root.findall(".//tei:body/tei:div[@corresp]", NS)

    chunks = []
    for div in divs_with_corresp:
        corresp = div.get("corresp")
        latin_sentences, enhg_sentences, rest_sentences = extract_div_text(div)
        latin_text = " ".join(latin_sentences)
        enhg_text  = " ".join(enhg_sentences)
        rest_text  = " ".join(rest_sentences)
        full_text  = " ".join(filter(None, [latin_text, enhg_text, rest_text]))

        if not full_text.strip():
            continue

        regest_text = regest_lookup.get(corresp, "")

        if regest_text:
            text_with_regest = f"[Summary: {regest_text}] {full_text}"
        else:
            text_with_regest = full_text

        chunks.append({
            "doc_id":            f"{letter_id}_{corresp}",
            "letter_id":         letter_id,
            "date":              date,
            "regest_id":         corresp,
            "regest_text":       regest_text,
            "latin_text":        latin_text,
            "enhg_text":         enhg_text,
            "rest_text":         rest_text,
            "text":              full_text,           # plain letter text only
            "text_with_regest":  text_with_regest,     # regest + letter text combined
            "n_sentences":       len(latin_sentences) + len(enhg_sentences) + len(rest_sentences),
        })

    return chunks


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    all_files = sorted(LETTERS_DIR.glob(GLOB_PATTERN))
    print(f"Found {len(all_files):,} letter files in {LETTERS_DIR}")

    if len(all_files) == 0:
        raise FileNotFoundError(
            f"No files matching '{GLOB_PATTERN}' found in {LETTERS_DIR}. "
            "Check LETTERS_DIR/GLOB_PATTERN at the top of the script."
        )

    # First pass: find which letters have regest structure
    print("Scanning corpus for letters with regest annotations...")
    regest_letter_files = []
    for fp in tqdm(all_files, desc="Scanning"):
        try:
            tree = etree.parse(str(fp))
            root = tree.getroot()
            if root.get("source", "") == "keine":
                continue
            if has_regest_structure(root):
                regest_letter_files.append(fp)
        except Exception:
            continue

    print(f"Letters with regest structure: {len(regest_letter_files):,} / {len(all_files):,}")

    if len(regest_letter_files) == 0:
        raise ValueError("No letters with regest structure found in this corpus.")

    rng = random.Random(SEED)
    n = min(N_SAMPLES, len(regest_letter_files))
    sampled_files = rng.sample(regest_letter_files, n)
    print(f"Randomly sampled {n} regest-bearing letters (seed={SEED})")

    all_chunks = []
    for fp in tqdm(sampled_files, desc="Chunking letters by regest"):
        chunks = extract_letter_chunks_with_regest(fp)
        all_chunks.extend(chunks)

    df = pd.DataFrame(all_chunks)
    print(f"\nLetters sampled: {n}")
    print(f"Total chunks produced: {len(df)}")
    print(f"Avg chunks per letter: {len(df) / df['letter_id'].nunique():.2f}")
    print(f"\nText length distribution (words, plain text):")
    print(df["text"].str.split().str.len().describe())
    print(f"\nText length distribution (words, with regest):")
    print(df["text_with_regest"].str.split().str.len().describe())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    print(df[["doc_id", "letter_id", "regest_id", "regest_text"]].head(5))


if __name__ == "__main__":
    main()