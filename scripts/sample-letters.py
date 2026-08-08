"""
Randomly sample n letters, extract their text (split by language tag plus a combined full_text)
"""

import os
import random
from pathlib import Path

import pandas as pd
from lxml import etree
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────
LETTERS_DIR = Path("../bullinger-korpus-tei/data/letters")   # folder containing Bullinger TEI-XML 
OUTPUT_CSV  = Path("../data/lloom_pilot_sample.csv")

N_SAMPLES = 50
SEED      = 42          # fixed seed so the sample is reproducible
GLOB_PATTERN = "*.xml"

# ── TEI namespace / tag constants ───────────────────────────────────────────
NS = {"tei": "http://www.tei-c.org/ns/1.0"}
TEI_NS = "{http://www.tei-c.org/ns/1.0}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

SKIP_TAGS = {
    f"{TEI_NS}note",
    f"{TEI_NS}persName",
    f"{TEI_NS}placeName",
}
LB_TAG = f"{TEI_NS}lb"
P_TAG  = f"{TEI_NS}p"


# ── Text extraction helpers ─────────────────────────────────────────────────
def get_sentence_text(element, is_root=True):
    """Extract text from a <s> element, skipping notes/persNames/placeNames."""
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
    """Fallback extraction for letters without <s> tags, using <lb>/<p>."""
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


def extract_letter(filepath):
    """Parse one TEI-XML letter file and return a dict of extracted fields."""
    try:
        tree = etree.parse(str(filepath))
        root = tree.getroot()
    except Exception as e:
        print(f"Parse error: {filepath}: {e}")
        return None

    source = root.get("source", "")
    if source == "keine":
        return None

    letter_id = root.get(
        "{http://www.w3.org/XML/1998/namespace}id",
        os.path.splitext(os.path.basename(filepath))[0],
    )

    sent = root.find(".//tei:correspAction[@type='sent']/tei:date", NS)
    date = sent.get("when") if sent is not None else None

    lang_usage = {}
    for lang_el in root.findall(".//tei:langUsage/tei:language", NS):
        ident = lang_el.get("ident", "")
        usage = lang_el.get("usage", "0")
        try:
            lang_usage[ident] = int(usage)
        except ValueError:
            lang_usage[ident] = 0

    latin_sentences, enhg_sentences, rest_sentences = [], [], []
    latin_text, enhg_text, rest_text = "", "", ""

    s_tags = root.findall(".//tei:body//tei:s", NS)

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

        latin_text = " ".join(latin_sentences)
        enhg_text  = " ".join(enhg_sentences)
        rest_text  = " ".join(rest_sentences)

    else:
        # Fallback: no per-sentence language tags, use whole-body text
        # assign it based on which language dominates via langUsage.
        body = root.find(".//tei:body", NS)
        full_text_parts = []
        if body is not None:
            walk_body_lb(body, full_text_parts)
        full_text = " ".join(full_text_parts)

        la_pct = lang_usage.get("la", 0)
        de_pct = lang_usage.get("de", 0)

        if la_pct >= de_pct:
            latin_text = full_text
        else:
            enhg_text = full_text

    return {
        "doc_id":            letter_id,
        "date":              date,
        "latin_text":        latin_text,
        "enhg_text":         enhg_text,
        "rest_text":         rest_text,
        "text":              " ".join(filter(None, [latin_text, enhg_text, rest_text])),
        "source":            source,
        "n_latin_sentences": len(latin_sentences),
        "n_enhg_sentences":  len(enhg_sentences),
        "n_rest_sentences":  len(rest_sentences),
    }


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    all_files = sorted(LETTERS_DIR.glob(GLOB_PATTERN))
    print(f"Found {len(all_files):,} letter files in {LETTERS_DIR}")

    if len(all_files) == 0:
        raise FileNotFoundError(
            f"No files matching '{GLOB_PATTERN}' found in {LETTERS_DIR}. "
            "Check LETTERS_DIR/GLOB_PATTERN at the top of the script."
        )

    rng = random.Random(SEED)
    n = min(N_SAMPLES, len(all_files))
    sampled_files = rng.sample(all_files, n)
    print(f"Randomly sampled {n} files (seed={SEED})")

    records = []
    skipped = 0
    for fp in tqdm(sampled_files, desc="Extracting letters"):
        rec = extract_letter(fp)
        if rec and rec["text"].strip():
            records.append(rec)
        else:
            skipped += 1

    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} letters ({skipped} skipped: parse error / empty text)")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")
    print(df[["doc_id", "date", "n_latin_sentences", "n_enhg_sentences"]].head(10))


if __name__ == "__main__":
    main()