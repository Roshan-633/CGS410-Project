"""
Step 2: Feature Extraction from CoNLL-U treebank files.
Reads each language's .conllu file, parses dependency trees,
and computes a rich feature vector for each language.

Features computed (25 total):
  — Tree-level structural features
  — Head-direction features (overall + per-POS)
  — Part-of-speech distribution
  — Argument order indicators
  — Projectivity
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

DATA_DIR  = "./data"
OUT_DIR   = "./results"
os.makedirs(OUT_DIR, exist_ok=True)

# Universal POS tags (17 tags from UD)
UPOS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
             "AUX",  "CCONJ","SCONJ","PART","NUM", "PROPN","PUNCT",
             "SYM",  "X",    "INTJ"]

MAX_SENTENCES = 5000   # cap per language (speeds things up, still robust)


# ─────────────────────────────────────────────
# CoNLL-U parser
# ─────────────────────────────────────────────

def parse_conllu(filepath, max_sents=MAX_SENTENCES):
    """
    Generator: yields one sentence at a time as a list of token dicts.
    Skips multi-word tokens (lines like '1-2') and empty nodes ('1.1').
    """
    sentence = []
    count = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if sentence:
                    yield sentence
                    count += 1
                    if count >= max_sents:
                        return
                    sentence = []
                continue
            parts = line.split("\t")
            if len(parts) != 10:
                continue
            idx = parts[0]
            # Skip multi-word tokens and empty nodes
            if "-" in idx or "." in idx:
                continue
            try:
                token = {
                    "id":     int(parts[0]),
                    "form":   parts[1],
                    "lemma":  parts[2],
                    "upos":   parts[3],
                    "head":   int(parts[6]) if parts[6] != "_" else 0,
                    "deprel": parts[7].split(":")[0],   # use coarse deprel
                }
                sentence.append(token)
            except ValueError:
                continue


# ─────────────────────────────────────────────
# Tree utilities
# ─────────────────────────────────────────────

def build_children(sentence):
    """Return dict: head_id → list of child token ids."""
    children = defaultdict(list)
    for tok in sentence:
        children[tok["head"]].append(tok["id"])
    return children


def tree_depth(tok_id, children, memo={}):
    """Recursive depth calculation with memoization."""
    if tok_id not in memo:
        if not children[tok_id]:
            memo[tok_id] = 0
        else:
            memo[tok_id] = 1 + max(tree_depth(c, children, memo)
                                   for c in children[tok_id])
    return memo[tok_id]


def get_token_depths(sentence, children):
    """Return depth of every token from the root."""
    root_id = next((t["id"] for t in sentence if t["head"] == 0), None)
    if root_id is None:
        return []
    depths = {}

    def dfs(node, d):
        depths[node] = d
        for child in children[node]:
            dfs(child, d + 1)

    dfs(root_id, 0)
    return [depths.get(t["id"], 0) for t in sentence]


def is_projective(sentence):
    """
    Check if the dependency arcs in a sentence are projective.
    An arc (i→j) crosses arc (k→l) if i<k<j<l or k<i<l<j.
    """
    arcs = [(t["head"], t["id"]) for t in sentence if t["head"] != 0]
    for i, (h1, d1) in enumerate(arcs):
        l1, r1 = min(h1, d1), max(h1, d1)
        for h2, d2 in arcs[i+1:]:
            l2, r2 = min(h2, d2), max(h2, d2)
            if l1 < l2 < r1 < r2 or l2 < l1 < r2 < r1:
                return False
    return True


# ─────────────────────────────────────────────
# Feature extraction for one sentence
# ─────────────────────────────────────────────

def extract_sentence_features(sentence):
    """
    Returns a dict of raw measurements for one sentence.
    These are later aggregated (mean/proportion) across all sentences.
    """
    n = len(sentence)
    if n < 2:
        return None

    id_to_tok = {t["id"]: t for t in sentence}
    children   = build_children(sentence)

    # ── 1. Dependency lengths ──────────────────────────────
    dep_lengths = [abs(t["id"] - t["head"])
                   for t in sentence if t["head"] != 0]
    mean_dep_len = np.mean(dep_lengths) if dep_lengths else 0
    std_dep_len  = np.std(dep_lengths)  if dep_lengths else 0

    # ── 2. Head direction (right-headed = head index < dep index) ──
    right_headed = [1 if t["head"] < t["id"] and t["head"] != 0 else 0
                    for t in sentence if t["head"] != 0]
    head_dir_ratio = np.mean(right_headed) if right_headed else 0.5

    # Per-POS head direction ratios
    pos_dir = defaultdict(list)
    for t in sentence:
        if t["head"] != 0:
            is_right = 1 if t["head"] < t["id"] else 0
            pos_dir[t["upos"]].append(is_right)
    adp_dir  = np.mean(pos_dir.get("ADP",  [0.5]))  # preposition vs postposition
    verb_dir = np.mean(pos_dir.get("VERB", [0.5]))
    noun_dir = np.mean(pos_dir.get("NOUN", [0.5]))
    adj_dir  = np.mean(pos_dir.get("ADJ",  [0.5]))

    # ── 3. Tree depth ─────────────────────────────────────
    depths     = get_token_depths(sentence, children)
    mean_depth = np.mean(depths) if depths else 0
    max_depth  = max(depths)     if depths else 0

    # ── 4. Branching factor (arity) ───────────────────────
    num_children = [len(children[t["id"]]) for t in sentence]
    mean_arity   = np.mean(num_children)
    max_arity    = max(num_children)

    # ── 5. POS distribution ───────────────────────────────
    pos_counts = defaultdict(int)
    for t in sentence:
        pos_counts[t["upos"]] += 1
    pos_freq = {p: pos_counts[p] / n for p in UPOS_TAGS}

    # ── 6. Root POS ───────────────────────────────────────
    root_is_verb = int(any(t["head"] == 0 and t["upos"] == "VERB"
                           for t in sentence))

    # ── 7. Argument order (SVO indicators) ────────────────
    # Find verb positions; check if nsubj/obj precede or follow
    verb_positions  = {t["id"]: t["id"] for t in sentence if t["upos"] == "VERB"}
    nsubj_before_v  = []
    obj_before_v    = []
    aux_before_v    = []
    for t in sentence:
        head_tok = id_to_tok.get(t["head"])
        if head_tok and head_tok["upos"] == "VERB":
            before = 1 if t["id"] < t["head"] else 0
            if t["deprel"] == "nsubj":
                nsubj_before_v.append(before)
            elif t["deprel"] == "obj":
                obj_before_v.append(before)
            elif t["deprel"] == "aux":
                aux_before_v.append(before)

    # ── 8. Projectivity ───────────────────────────────────
    projective = int(is_projective(sentence))

    # ── 9. Non-projective arc count ───────────────────────
    arcs = [(t["head"], t["id"]) for t in sentence if t["head"] != 0]
    nonproj_count = 0
    for i, (h1, d1) in enumerate(arcs):
        l1, r1 = min(h1, d1), max(h1, d1)
        for h2, d2 in arcs[i+1:]:
            l2, r2 = min(h2, d2), max(h2, d2)
            if l1 < l2 < r1 < r2 or l2 < l1 < r2 < r1:
                nonproj_count += 1

    return {
        "mean_dep_len":     mean_dep_len,
        "std_dep_len":      std_dep_len,
        "head_dir_ratio":   head_dir_ratio,
        "adp_dir":          adp_dir,
        "verb_dir":         verb_dir,
        "noun_dir":         noun_dir,
        "adj_dir":          adj_dir,
        "mean_depth":       mean_depth,
        "max_depth":        max_depth,
        "mean_arity":       mean_arity,
        "max_arity":        max_arity,
        "root_is_verb":     root_is_verb,
        "nsubj_before_v":   np.mean(nsubj_before_v) if nsubj_before_v else 0.5,
        "obj_before_v":     np.mean(obj_before_v)   if obj_before_v   else 0.5,
        "aux_before_v":     np.mean(aux_before_v)   if aux_before_v   else 0.5,
        "projective":       projective,
        "nonproj_count":    nonproj_count,
        "sent_length":      n,
        **{f"pos_{p}": pos_freq[p] for p in UPOS_TAGS},
    }


# ─────────────────────────────────────────────
# Aggregate across all sentences in a language
# ─────────────────────────────────────────────

def extract_language_features(filepath, language):
    """Aggregate sentence-level features into a single language feature vector."""
    records = []
    for sent in parse_conllu(filepath):
        feats = extract_sentence_features(sent)
        if feats:
            records.append(feats)

    if not records:
        print(f"  [WARN] No records for {language}")
        return None

    df = pd.DataFrame(records)
    agg = df.mean()   # mean over all sentences
    agg["n_sentences"] = len(records)
    agg["language"]    = language
    print(f"  [OK] {language}: {len(records)} sentences, "
          f"MDL={agg['mean_dep_len']:.2f}, "
          f"HDR={agg['head_dir_ratio']:.2f}, "
          f"depth={agg['mean_depth']:.2f}")
    return agg


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # Load metadata (word orders)
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    rows = []
    for lang in sorted(os.listdir(DATA_DIR)):
        if not lang.endswith(".conllu"):
            continue
        language = lang.replace(".conllu", "")
        filepath = os.path.join(DATA_DIR, lang)
        print(f"\nProcessing: {language}")
        feats = extract_language_features(filepath, language)
        if feats is not None:
            feats["word_order"] = metadata.get(language, {}).get("word_order", "UNK")
            rows.append(feats)

    df = pd.DataFrame(rows).set_index("language")

    # Drop non-feature columns before saving feature matrix
    info_cols = ["n_sentences", "word_order"]
    feature_df = df.drop(columns=info_cols)
    meta_df    = df[info_cols]

    feature_df.to_csv(os.path.join(OUT_DIR, "features.csv"))
    meta_df.to_csv(   os.path.join(OUT_DIR, "metadata.csv"))

    print(f"\n[DONE] Feature matrix: {feature_df.shape}")
    print(f"Features: {list(feature_df.columns)}")
    print(f"\nSaved to: {os.path.abspath(OUT_DIR)}/features.csv")


if __name__ == "__main__":
    main()
