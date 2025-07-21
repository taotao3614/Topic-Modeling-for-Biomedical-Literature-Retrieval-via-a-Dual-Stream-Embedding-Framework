# Topic Modeling for Biomedical Literature Retrieval via a Dual-Stream Embedding Framework

## Project Overview

This repository contains the original data and code for the paper "Topic Modeling for Biomedical Literature Retrieval via a Dual-Stream Embedding Framework." The project aims to improve topic modelling in biomedical literature via a dual-stream embedding mechanism, integrating both unstructured text and structured biomedical knowledge (MeSH terms). It includes the full experimental pipeline, ablation studies, main results, and reproducible code and data for peer review and research reproducibility.

---

## Directory Structure

```
experiment/
│
├── experiment result analysis.md      # Experimental design and results analysis (EN & CN)
├── reference.png                     # Reference model result chart
│
├── ablation/                         # Ablation experiments
│   ├── data_process_only/            # Data processing ablation (various text combinations)
│   │   ├── full.ipynb, text.ipynb, abmesh.ipynb, abstract.ipynb
│   │   └── topics_with_titles.csv    # Topic-title mapping
│   ├── with_model/                   # Model replacement ablation
│   │   └── abstract+abstract.ipynb
│   ├── raw.csv                       # Raw data for ablation
│   └── data_check.py                 # Data checking script
│
└── dual-stream/                      # Main dual-stream embedding experiments
    ├── data/                         # Metadata and analysis scripts
    │   ├── raw data.csv              # 3,000 PubMed biomedical records
    │   ├── tm-viewer.ipynb           # Topic visualization
    │   └── data_prepare.py           # Data preprocessing & embedding generation
    ├── Main_Experiment_Report_CN.md  # Main experiment report (Chinese)
    ├── Main_Experiment_Report_EN.md  # Main experiment report (English)
    ├── results/                      # Detailed results (various models/parameters)
    ├── st+st-full/                   # Dual pubmedbert model results
    ├── st+bio-full/                  # pubmedbert+biobert hybrid model results
    ├── bio+bio-full/                 # Dual biobert model results
    ├── basic-model-full/             # Baseline model results (all-MiniLM-L6-v2)
    └── data_prepare.py               # Data processing & embedding script
```

---

## Key Content Description

### 1. Metadata (`experiment/dual-stream/data/raw data.csv`)
- Contains 3,000 PubMed biomedical literature records, including title, abstract, MeSH terms, year, PMID, etc.
- Ready for direct use in topic modelling experiments.

### 2. Ablation Experiments (`experiment/ablation/`)
- `data_process_only/`: Ablation on different text combinations (full, text, abmesh, abstract) to analyze their impact on topic modelling.
- `with_model/`: Model replacement ablation to analyze the effect of different pre-trained models.
- `raw.csv`: Raw data for ablation studies.

### 3. Dual-Stream Embedding Experiments (`experiment/dual-stream/`)
- `data_prepare.py`: Implements the dual-stream embedding mechanism with parameterized fusion (α for text/MeSH, β for major/minor MeSH weighting).
- `st+st-full/`, `st+bio-full/`, `bio+bio-full/`, `basic-model-full/`: Results for different model combinations, including vector files (.npy), detailed results (.txt), visualizations (.png), and analysis notebooks (.ipynb).
- `results/`: Detailed results under various models and parameters for reproducibility and comparison.

### 4. Experiment Reports & Analysis
- `Main_Experiment_Report_CN.md` / `EN.md`: Main experiment summary, including baseline comparison, metric improvements, and parameter sensitivity analysis.
- `experiment result analysis.md`: Detailed experimental design, methodology, parameter explanation, and results analysis.

---

## Quick Start

1. Install dependencies (Python 3.8+ recommended; required packages include `transformers`, `sentence-transformers`, `umap-learn`, `hdbscan`, etc.).
2. Run `experiment/dual-stream/data_prepare.py` for data preprocessing and embedding generation.
3. Use the notebooks (e.g., `result.ipynb` in each experiment folder) for result analysis and visualization.

---

## Main Innovations

- **Dual-Stream Embedding Mechanism:** Integrates text and MeSH streams via parameterized weighted fusion, improving topic coherence and diversity.
- **Two-Layer Weighting:** Flexible adjustment of major/minor MeSH weighting (β) and text/MeSH weighting (α).
- **Comprehensive Ablation:** Systematic analysis of text combinations and model choices on topic modelling performance.
- **High Reproducibility:** All data, scripts, and results are provided for full reproducibility and peer verification.

---

## Authors & Contact

**Topic Modeling for Biomedical Literature Retrieval via a Dual-Stream Embedding Framework**

**Authors:**  
Tao Tao¹, Eiko Takaoka²

**Affiliations:**  
¹ Green Science and Engineering Division, Graduate School of Science and Technology, Sophia University, Tokyo, Japan  
² Department of Information and Communication Sciences, Faculty of Science and Technology, Sophia University, Tokyo, Japan

**Corresponding author:**  
Tao Tao  
Email: taotao3614@eagle.sophia.ac.jp  
Address: Green Science and Engineering Division, Graduate School of Science and Technology, Sophia University, 7-1 Kioicho, Chiyoda-ku, Tokyo 102--8554, Japan

**Keywords:**  
Topic modelling; Dual-stream embedding; BERTopic; PubMed; MeSH; Biomedical text mining; Transformer models; LLM interpretability; Biomedical information retrieval; Semantic clustering
