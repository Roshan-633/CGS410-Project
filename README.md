
# 1. Install dependencies (once)
pip install -r requirements.txt
  
# 2. Download treebank data (~30 languages from UD GitHub)
python 01_download_data.py : downloaded Universal Dependencies treebanks and saved language-level word-order metadata.
  
# 3. Parse each treebank and compute features
python 02_extract_features.py : parsed CoNLL-U files and computed dependency-based structural features for each language.
  
# 4. Run ALL clustering + generate ALL figures
python 03_clustering_analysis.py : selected the final feature set, standardized features, applied PCA, t-SNE, K-Means, hierarchical clustering, and DBSCAN, and saved figures and tables.

# 5. Run cluster evaluation and purity analysis
python 04_purity_analysis.py : computed external evaluation metrics including ARI, NMI, homogeneity, completeness, V-measure, cluster purity, confusion matrices, radar chart, and per-language assignment tables.
