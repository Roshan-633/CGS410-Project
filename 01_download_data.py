"""
Step 1: Download Universal Dependencies (UD) treebank data.
Run this first. It downloads .conllu files for ~36 languages into ./data/

CHANGE LOG:
  - Added 3 VSO languages: Breton, Coptic, Ancient Hebrew  → total VSO: 6
  - Added 3 SOV languages: Tamil, Urdu, Amharic            → total SOV: 10
  - Final breakdown: SVO ~16, SOV ~10, VSO ~6 (much more balanced)
"""

import os
import json
import requests
import time

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Language → (GitHub repo name, train filename, word order)
#
# Word orders from WALS / established typological literature.
# Only using SVO / SOV / VSO — all clean, unambiguous classifications.
# ─────────────────────────────────────────────────────────────────────────────

TREEBANKS = {

    # ── SVO (head-initial, subject before verb, verb before object) ────────────
    "English":        ("UD_English-EWT",           "en_ewt-ud-train.conllu",            "SVO"),
    "Spanish":        ("UD_Spanish-GSD",            "es_gsd-ud-train.conllu",            "SVO"),
    "French":         ("UD_French-GSD",             "fr_gsd-ud-train.conllu",            "SVO"),
    "Italian":        ("UD_Italian-ISDT",           "it_isdt-ud-train.conllu",           "SVO"),
    "Portuguese":     ("UD_Portuguese-GSD",         "pt_gsd-ud-train.conllu",            "SVO"),
    "Romanian":       ("UD_Romanian-RRT",           "ro_rrt-ud-train.conllu",            "SVO"),
    "Catalan":        ("UD_Catalan-AnCora",         "ca_ancora-ud-train.conllu",         "SVO"),
    "Swedish":        ("UD_Swedish-Talbanken",      "sv_talbanken-ud-train.conllu",      "SVO"),
    "Norwegian":      ("UD_Norwegian-Bokmaal",      "nb_bokmaal-ud-train.conllu",        "SVO"),
    "Danish":         ("UD_Danish-DDT",             "da_ddt-ud-train.conllu",            "SVO"),
    "Russian":        ("UD_Russian-SynTagRus",      "ru_syntagrus-ud-train.conllu",      "SVO"),
    "Czech":          ("UD_Czech-PDT",              "cs_pdt-ud-train.conllu",            "SVO"),
    "Polish":         ("UD_Polish-LFG",             "pl_lfg-ud-train.conllu",            "SVO"),
    "Bulgarian":      ("UD_Bulgarian-BTB",          "bg_btb-ud-train.conllu",            "SVO"),
    "Ukrainian":      ("UD_Ukrainian-IU",           "uk_iu-ud-train.conllu",             "SVO"),
    "Finnish":        ("UD_Finnish-TDT",            "fi_tdt-ud-train.conllu",            "SVO"),
    "Hebrew":         ("UD_Hebrew-HTB",             "he_htb-ud-train.conllu",            "SVO"),
    "Chinese":        ("UD_Chinese-GSD",            "zh_gsd-ud-train.conllu",            "SVO"),
    "Indonesian":     ("UD_Indonesian-GSD",         "id_gsd-ud-train.conllu",            "SVO"),

    # ── SOV (head-final, verb at end) ──────────────────────────────────────────
    "Turkish":        ("UD_Turkish-IMST",           "tr_imst-ud-train.conllu",           "SOV"),
    "Persian":        ("UD_Persian-Seraji",         "fa_seraji-ud-train.conllu",         "SOV"),
    "Hindi":          ("UD_Hindi-HDTB",             "hi_hdtb-ud-train.conllu",           "SOV"),
    "Japanese":       ("UD_Japanese-GSD",           "ja_gsd-ud-train.conllu",            "SOV"),
    "Korean":         ("UD_Korean-GSD",             "ko_gsd-ud-train.conllu",            "SOV"),
    "Basque":         ("UD_Basque-BDT",             "eu_bdt-ud-train.conllu",            "SOV"),
    "Latin":          ("UD_Latin-ITTB",             "la_ittb-ud-train.conllu",           "SOV"),
    # NEW SOV ──────────────────────────────────────────────────────────────────
    # Urdu: closely related to Hindi but independent treebank, different script
    # Good test: do Hindi and Urdu cluster together despite surface differences?
    "Urdu":           ("UD_Urdu-UDTB",              "ur_udtb-ud-train.conllu",           "SOV"),
    # Tamil: Dravidian — entirely different family from all other SOV langs above
    # Adds typological diversity beyond Indo-European and Turkic
    "Tamil":          ("UD_Tamil-TTB",              "ta_ttb-ud-train.conllu",            "SOV"),
    # Amharic: Semitic but SOV (unlike Arabic which is VSO)
    # Demonstrates that Semitic ≠ always VSO — purely structural features should show this
    "Amharic":        ("UD_Amharic-ATT",            "am_att-ud-train.conllu",            "SOV"),

    # ── VSO (verb-initial, verb before subject and object) ────────────────────
    "Arabic":         ("UD_Arabic-PADT",            "ar_padt-ud-train.conllu",           "VSO"),
    "Irish":          ("UD_Irish-IDT",              "ga_idt-ud-train.conllu",            "VSO"),
    "Welsh":          ("UD_Welsh-CCG",              "cy_ccg-ud-train.conllu",            "VSO"),
    # NEW VSO ──────────────────────────────────────────────────────────────────
    # Breton: Celtic VSO, same family as Welsh/Irish — confirms Celtic VSO pattern
    "Breton":         ("UD_Breton-KEB",             "br_keb-ud-train.conllu",            "VSO"),
    # Coptic: ancient Egyptian language, VSO, non-Indo-European
    # Adds a completely different language family to the VSO class
    "Coptic":         ("UD_Coptic-Scriptorium",     "cop_scriptorium-ud-train.conllu",   "VSO"),
    # Ancient Hebrew: Biblical Hebrew, VSO Semitic (contrasts with Amharic which is SOV Semitic)
    # Key test: can features distinguish VSO-Semitic (Arabic, Hebrew) from SOV-Semitic (Amharic)?
    "AncientHebrew":  ("UD_Ancient_Hebrew-PTNK",   "hbo_ptnk-ud-train.conllu",          "VSO"),

}

# ─────────────────────────────────────────────────────────────────────────────
# Note on languages excluded deliberately:
#   German, Dutch  — WALS lists both as having two dominant orders (SVO+SOV);
#                    labelling them SOV would add noise to the ground truth
#   Hungarian      — WALS lists as no dominant order; too risky to label
#   Maltese        — VSO in classical register but heavily mixed with SVO;
#                    ambiguous ground truth
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://raw.githubusercontent.com/UniversalDependencies"


def download_file(language, repo, filename, word_order):
    """Try master → main → dev branches."""
    save_path = os.path.join(DATA_DIR, f"{language}.conllu")

    if os.path.exists(save_path):
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"  [SKIP] {language} already downloaded ({size_mb:.1f} MB).")
        return True

    for branch in ["master", "main", "dev"]:
        url = f"{BASE_URL}/{repo}/{branch}/{filename}"
        try:
            print(f"  Trying {url} ...")
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                size_mb = os.path.getsize(save_path) / 1e6
                print(f"  [OK] {language} ({word_order}) — {size_mb:.1f} MB")
                return True
        except Exception as e:
            print(f"  [WARN] {branch}: {e}")
        time.sleep(0.5)

    print(f"  [FAIL] {language} — could not download from any branch.")
    return False


def save_metadata():
    meta = {lang: {"repo": repo, "file": fn, "word_order": wo}
            for lang, (repo, fn, wo) in TREEBANKS.items()}
    with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n[OK] metadata.json saved.")


def print_summary(results):
    from collections import Counter
    wo_counts = Counter()
    print("\n" + "=" * 55)
    print("DOWNLOAD SUMMARY")
    print("=" * 55)
    for lang, (success, wo) in results.items():
        status = "OK  " if success else "FAIL"
        print(f"  [{status}] {lang:<20} [{wo}]")
        if success:
            wo_counts[wo] += 1

    print("\n  Final class distribution (successfully downloaded):")
    for wo in ["SVO", "SOV", "VSO"]:
        bar = "█" * wo_counts[wo]
        print(f"    {wo}: {bar} {wo_counts[wo]}")
    total = sum(wo_counts.values())
    print(f"\n  Total: {total} languages")


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading Universal Dependencies treebanks")
    print(f"Target: {len(TREEBANKS)} languages")
    print("=" * 60)

    results = {}
    for lang, (repo, fn, wo) in TREEBANKS.items():
        print(f"\n→ {lang} ({wo})")
        ok = download_file(lang, repo, fn, wo)
        results[lang] = (ok, wo)

    save_metadata()
    print_summary(results)
    print(f"\nData saved in: {os.path.abspath(DATA_DIR)}")
