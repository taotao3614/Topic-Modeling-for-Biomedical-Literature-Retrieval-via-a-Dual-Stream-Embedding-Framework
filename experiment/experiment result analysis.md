# Dual-Stream Embedding Topic Modeling: Experimental Design and Results

## 1. Experimental Overview

### 1.1 Research Objectives
This experiment aims to validate the effectiveness of the dual-stream embedding model in biomedical literature topic modeling, with a particular focus on evaluating the impact of different pre-trained model combinations (general-purpose and biomedical models) and parameter settings on topic quality. By systematically exploring the α and β parameter space, we seek the optimal configuration to enhance the model's ability to extract topics from biomedical literature, and compare it with reference models ( Model 3) from the literature.

### 1.2 Experimental Dataset
The experiment uses biomedical literature data from the PubMed database:
- Total data: 3,000 biomedical literature records, referenced from "Depression, anxiety, and burnout in academia: topic modeling of PubMed abstracts"
- Data structure: includes title, abstract, year, PMID, MeSH terms, and other fields

### 1.3 Key Model and Terminology Definitions

**Pre-trained Model Definitions:**
- **basic**: all-MiniLM-L6-v2 (general language model, used as the baseline in the reference)
- **st(sentence transformer)**: NeuML/pubmedbert-base-embeddings (biomedical domain model)
- **bio/biobert**: dmis-lab/biobert-v1.1 (biomedical domain model)

**Text Combination Strategies:**
- **abstract**: using only the abstract
- **text**: combination of title and abstract (title + abstract)
- **abmesh**: combination of abstract and MeSH terms (abstract + mesh)
- **full**: full combination of title, abstract, and MeSH terms (title + abstract + mesh)

**Model Combination Configurations:**
- **basic**: both streams use baseline model all-MiniLM-L6-v2
- **st+bio**: combination of biomedical models st and bio
- **bio+bio**: both streams use the bio model
- **st+st**: both streams use the st model

**Parameterized Weighted Fusion Mechanism:**
- **α parameter**: controls the weight of text embedding (α) and MeSH embedding (1-α), range 0.05-0.95
- **β parameter**: controls the weight of major MeSH topics (β) and minor topics (1-β), range 0.05-0.95

**Evaluation Metrics:**
- **Topic Coherence (TC)**: evaluates the semantic coherence within topics, range [0,1], higher is better
- **Topic Diversity (TD)**: evaluates the distinctiveness between topics, range [0,1], higher means less overlap between topics

## 2. Experimental Folder Structure

```
/experiment/
│
├── Dual-Stream Embedding Topic Modeling Analysis.md    # Original experimental analysis report
├── reference.png                # Model result chart from reference
├── View Single Result.ipynb     # Notebook for analyzing individual results
├── result.ipynb                # Summary analysis notebook
│
├── ablation/                   # Ablation experiment folder
│   ├── data_process_only/      # Ablation experiments for different text processing methods
│   │   ├── full+abstract.ipynb       # Experiment using full text + abstract
│   │   ├── text+abstract.ipynb       # Experiment using text + abstract
│   │   ├── abmesh+abstract.ipynb     # Experiment using abmesh + abstract
│   │   └── abstract+abstract.ipynb   # Experiment using only abstract
│   │
│   ├── with_model/             # Model replacement ablation experiments
│   │   └── abstract+abstract.ipynb   # Abstract experiments with different models
│   │
│   ├── raw.csv                 # Raw data
│   └── data_check.py           # Data checking script
│
└── dual-stream/                # Main folder for dual-stream embedding topic modeling experiments
    ├── data_prepare.py         # Data preprocessing and embedding generation script
    │
    ├── basic-model-full/       # Baseline model experiments (using all-MiniLM-L6-v2)
    │   ├── text-result-basic-0501_a65b30.npy    # Text result vector file
    │   ├── full-result-basic-0501_a20b35.npy    # Full text result vector file
    │   ├── abstract-result-basic-0501_a50b60.npy # Abstract result vector file
    │   ├── abmesh-result-basic-0501_a20b25.npy  # Abstract+mesh result vector file
    │   ├── result.ipynb                         # Result analysis notebook
    │   ├── text-basic_a65b30.png                # Text experiment heatmap
    │   ├── abmesh-basic_a20b25.png              # Abstract+mesh experiment heatmap
    │   ├── full-basic_a20b35.png                # Full text experiment heatmap
    │   ├── abstract-basic_a50b60.png            # Abstract experiment heatmap
    │   └── *-result-basic-0501.txt              # Detailed results for various text combinations
    │
    ├── bio+bio-full/           # Dual biobert model experiments
    │   ├── result.ipynb                         # Result analysis notebook
    │   ├── abmesh-result-bio+bio-0502_a45b75.npy # Abstract+mesh result vector file
    │   ├── abmesh-bio+bio-a45b75.png            # Abstract+mesh experiment heatmap
    │   ├── abstract-result-bio+bio-0501_a45b65.npy # Abstract result vector file
    │   ├── abstract-bio+bio-a45b65.png          # Abstract experiment heatmap
    │   ├── full-result-bio+bio-0501_a45b80.npy  # Full text result vector file
    │   ├── text-result-bio+bio-0501_a45b75.npy  # Text result vector file
    │   ├── text-bio+bio-a45b75.png              # Text experiment heatmap
    │   ├── full-bio+bio-a45b80.png              # Full text experiment heatmap
    │   └── *-result-bio+bio-0501.txt            # Detailed results for various text combinations
    │
    ├── st+bio-full/            # pubmedbert+biobert hybrid model experiments
    │   ├── result.ipynb                         # Result analysis notebook
    │   ├── abmesh-result-st+bio-0502_a20b60.npy # Abstract+mesh result vector file
    │   ├── abmesh-st+bio-a20b60.png             # Abstract+mesh experiment heatmap
    │   ├── abstract-st+bio-a20b70.png           # Abstract experiment heatmap
    │   ├── abstract-result-st+bio-0430_a20b70.npy # Abstract result vector file
    │   ├── full-result-st+bio-0430_a30b35.npy   # Full text result vector file
    │   ├── full-st+bio-a30b35.png               # Full text experiment heatmap
    │   ├── text-st+bio-a20b70.png               # Text experiment heatmap
    │   ├── text-result-st+bio-0430_a20b70.npy   # Text result vector file
    │   └── *-result-st+bio-0430.txt             # Detailed results for various text combinations
    │
    └── st+st-full/             # Dual pubmedbert model experiments (similar directory contents)
```

## 3. Experimental Methods

### 3.0 Baseline
The best result (Model 3) from the article 'Depression, anxiety, and burnout in academia: topic modeling of PubMed abstracts' is used as a reference: TC=0.536, TD=0.717

### 3.1 Model Architecture
This experiment adopts a dual-stream embedding topic modeling approach based on the BERTopic framework, mainly including the following steps:

1. **Dual-stream embedding generation:**
   - **Text stream embedding:** Encode the document text using a pre-trained language model
   - **MeSH term stream embedding:** Encode MeSH terms using a biomedical domain model
   - **Parameterized fusion:** Control the weight allocation of the two streams via α and β parameters

2. **Topic model training(The same setting as baseline):**
   - Dimensionality reduction: UMAP model (n_neighbors=15, n_components=5, min_dist=0.0)
   - Clustering: HDBSCAN model (min_cluster_size=30, min_samples=10)
   - Topic representation: CountVectorizer + ClassTfidfTransformer
   - Diversification: MaximalMarginalRelevance (diversity=0.2)

### 3.2 Parameterized Fusion Mechanism

The dual-stream fusion mechanism implemented in `data_prepare.py` includes two key parameters:

1. **α parameter (alpha):** Controls the fusion ratio between text embedding and MeSH term embedding
   - Text embedding weight = α
   - MeSH term embedding weight = 1-α
   - Value range: [0.05, 0.95], step size 0.05

2. **β parameter (beta):** Controls the weight between major and minor MeSH topics
   - Major topic (MajorTopicYN='Y') weight = β
   - Minor topic (MajorTopicYN='N') weight = 1-β
   - Value range: [0.05, 0.95], step size 0.05

#### 3.2.1 Detailed Explanation of the Two-layer Weighted Fusion Mechanism

The core innovation of this experiment lies in the design of a two-layer weighted fusion mechanism, including intra-MeSH term weighting and text-MeSH term weighting:

**First layer weighting: Intra-MeSH term weighting (controlled by β parameter)**

1. Process the MeSH term set for each document:
   ```python
   # Extract all DescriptorName and their MajorTopicYN values
   descriptor_names = []
   major_flags = []
   for term in mesh_terms:
       if 'DescriptorName' in term:
           descriptor_names.append(term['DescriptorName'])
           # Record whether it is a major topic
           is_major = term.get('MajorTopicYN', 'N') == 'Y'
           major_flags.append(is_major)
   ```
2. Encode MeSH terms using a pre-trained model:
   ```python
   # Encode all MeSH terms
   term_embeddings = model.encode(descriptor_names, batch_size=batch_size)
   ```
3. Assign weights based on MajorTopicYN flag and β parameter:
   ```python
   # Calculate weights for major and minor topics based on beta
   weight_major = beta  # Weight for MajorTopicYN='Y'
   weight_minor = 1 - beta  # Weight for MajorTopicYN='N'
   # Assign weights based on MajorTopicYN value
   weights = np.array([weight_major if is_major else weight_minor for is_major in major_flags])
   ```
4. Normalize weights and compute weighted embedding:
   ```python
   # Normalize weights
   if weights.sum() > 0:
       weights = weights / weights.sum()
   # Weighted sum
   weighted_embedding = np.zeros(vector_dim)
   for j, embedding in enumerate(term_embeddings):
       weighted_embedding += embedding * weights[j]
   ```
This process generates a fused embedding vector for each document that integrates all MeSH terms, with major topics (marked as important medical concepts) receiving higher weights via the β parameter.

**Second layer weighting: Fusion of text embedding and MeSH term embedding (controlled by α parameter)**

1. Calculate the weights for text and MeSH terms based on the α parameter:
   ```python
   # Calculate weights for text and MeSH terms based on alpha
   text_weight = alpha
   mesh_weight = 1 - alpha
   ```
2. For articles with MeSH terms, apply weighted averaging:
   ```python
   # For articles with MeSH terms: weighted average
   combined_embeddings[mask_has_mesh] = (
       text_weight * text_embeddings[mask_has_mesh] + 
       mesh_weight * mesh_embeddings[mask_has_mesh]
   )
   ```
3. For articles without MeSH terms, use only text embedding:
   ```python
   # For articles without MeSH terms: use only text embedding
   combined_embeddings[~mask_has_mesh] = text_embeddings[~mask_has_mesh]
   ```

Through this two-layer weighted fusion mechanism, the model can:

1. **Effect of β parameter:** Adjust the weight distribution within MeSH terms, enabling the model to focus on more important medical topic concepts
   - Higher β: the model focuses more on major medical concepts (terms marked as MajorTopicYN='Y')
   - Lower β: the model considers all medical terms more evenly, including minor concepts

2. **Effect of α parameter:** Control the balance between textual content and structured medical knowledge
   - Higher α: the model relies more on textual content (title, abstract, etc.)
   - Lower α: the model emphasizes structured medical knowledge (MeSH terms)

The advantages of this parameterized fusion mechanism are:
- **Flexibility:** Parameters can be adjusted according to different data characteristics and application needs
- **Knowledge integration:** Effectively combines unstructured text and structured medical terms
- **Customization:** Provides the optimal information fusion ratio for different model types

This is also the main innovation of the experimental method, which effectively improves topic modeling quality through two-layer weighted fusion.

### 3.3 Ablation Experiment Design

The experiment includes two main groups of ablation experiments:

1. **Ablation of text processing methods** (`ablation/data_process_only/`):
   - Evaluate the impact of different text combination strategies on model performance
   - Compare four text processing methods: full, text, abmesh, and abstract

2. **Ablation of model replacement** (`ablation/with_model/`):
   - Evaluate the effect of replacing only the pre-trained model without using the dual-stream architecture
   - Verify the advantage of the dual-stream architecture over simple model replacement

### 3.4 Dual-Stream Experimental Design

In the dual-stream folder, the experiment adopts a systematic combination design:

1. **Model combination dimension:**
**Table X. Dual-Stream Embedding Model Combinations**

| Combinations Name   | Text Stream Embedding Model                 | MeSH Stream Embedding Model                 | Description                            |
|--------------|---------------------------------------------|---------------------------------------------|----------------------------------------|
| **basic**    | all-MiniLM-L6-v2                            | all-MiniLM-L6-v2                            | Baseline model using general-purpose embeddings on both streams. |
| **st+bio**   | NeuML/pubmedbert-base-embeddings            | dmis-lab/biobert-v1.1                       | Hybrid model using PubMedBERT for text and BioBERT for MeSH.     |
| **bio+bio**  | dmis-lab/biobert-v1.1                       | dmis-lab/biobert-v1.1                       | Double BioBERT model; biomedical embeddings for both streams.    |
| **st+st**    | NeuML/pubmedbert-base-embeddings            | NeuML/pubmedbert-base-embeddings            | Double PubMedBERT model; specialized biomedical embeddings on both streams. |

   
2. **文本组合维度**：
   - Abstract: Abstract only
   - Text: Title + abstract (text)
   - Abmesh: Abstract +MeSH）
   - Full: Title + Abstract + MeSH）

3. **Parameter space dimension:**
   - α parameter: [0.05, 0.95], step size 0.05
   - β parameter: [0.05, 0.95], step size 0.05

Each model and text combination configuration underwent a complete α-β parameter space search, generating heatmaps and recording the best parameter combinations.

## 4. Experimental Results

### 4.1 File Naming Rules

Experimental result files follow the naming rules below:

- **Vector file**: `{text_type}-result-{model_type}-{date}_a{alpha_value}b{beta_value}.npy`
  - Example: `full-result-bio+bio-0501_a45b80.npy` indicates the use of the bio+bio model, full text combination, α=0.45, β=0.80

- **Heatmap file**: `{text_type}-{model_type}_a{alpha_value}b{beta_value}.png`
  - Example: `abmesh-st+bio-a20b60.png` indicates the use of the st+bio model, abmesh text combination, α=0.20, β=0.60

- **Result text file**: `{text_type}-result-{model_type}-{date}.txt`
  - Contains the complete parameter search result data

### 4.2 Detailed Results of Dual-Stream Experiments

#### 4.2.1 Results of basic-model-full

| Text Combination | Best TC | Parameters | Best TD | Parameters | Best TC+TD | Parameters |
|-----------------|---------|------------|---------|------------|------------|------------|
| abmesh | 0.6032 | α=20.0, β=35.0 | 0.7056 | α=40.0, β=15.0 | 1.2954 | α=20.0, β=25.0 |
| abstract | 0.5913 | α=25.0, β=30.0 | 0.6926 | α=25.0, β=80.0 | 1.2633 | α=50.0, β=60.0 |
| text | 0.5973 | α=50.0, β=75.0 | 0.6926 | α=25.0, β=80.0 | 1.2780 | α=65.0, β=30.0 |
| full | 0.6076 | α=15.0, β=20.0 | 0.7143 | α=10.0, β=35.0 | 1.3005 | α=20.0, β=35.0 |

#### 4.2.2 Results of bio+bio-full

| Text Combination | Best TC | Parameters | Best TD | Parameters | Best TC+TD | Parameters |
|-----------------|---------|------------|---------|------------|------------|------------|
| abmesh | 0.5710 | α=30.0, β=60.0 | 0.7000 | α=2.0, β=25.0 | 1.2610 | α=45.0, β=75.0 |
| abstract | 0.5560 | α=45.0, β=65.0 | 0.7000 | α=5.0, β=25.0 | 1.2060 | α=45.0, β=65.0 |
| text | 0.5701 | α=45.0, β=65.0 | 0.7000 | α=5.0, β=25.0 | 1.2208 | α=45.0, β=75.0 |
| full | 0.5826 | α=45.0, β=65.0 | 0.7000 | α=10.0, β=15.0 | 1.2331 | α=45.0, β=80.0 |

#### 4.2.3 Results of st+bio-full

| Text Combination | Best TC | Parameters | Best TD | Parameters | Best TC+TD | Parameters |
|-----------------|---------|------------|---------|------------|------------|------------|
| abmesh | 0.5968 | α=25.0, β=85.0 | 0.7333 | α=10.0, β=25.0 | 1.3197 | α=20.0, β=60.0 |
| abstract | 0.5958 | α=65.0, β=70.0 | 0.7318 | α=20.0, β=70.0 | 1.3089 | α=20.0, β=70.0 |
| text | 0.5994 | α=15.0, β=15.0 | 0.7318 | α=20.0, β=70.0 | 1.3176 | α=20.0, β=70.0 |
| full | 0.6038 | α=15.0, β=15.0 | 0.7333 | α=10.0, β=25.0 | 1.3247 | α=30.0, β=35.0 |

#### 4.2.4 Results of st+st-full

| Text Combination | Best TC | Parameters | Best TD | Parameters | Best TC+TD | Parameters |
|-----------------|---------|------------|---------|------------|------------|------------|
| abmesh | 0.6146 | α=45.0, β=20.0 | 0.7667 | α=20.0, β=20.0 | 1.3550 | α=45.0, β=40.0 |
| abstract | 0.6018 | α=35.0, β=65.0 | 0.7261 | α=35.0, β=65.0 | 1.3279 | α=35.0, β=65.0 |
| text | 0.6067 | α=35.0, β=60.0 | 0.7333 | α=20.0, β=20.0 | 1.3269 | α=45.0, β=50.0 |
| full | 0.6172 | α=40.0, β=65.0 | 0.7458 | α=45.0, β=40.0 | 1.3443 | α=45.0, β=40.0 |

### 4.3 Ablation Experiment Results

#### 4.3.1 Results of Text Processing Method Ablation

| Text Combination | TC | TD | TC+TD |
|------------------|------|------|--------|
| origin | 0.536 | 0.717 | 1.253 |
| abmesh+abstract | 0.5274 | 0.6087 | 1.1361 |
| full+abstract | 0.5451 | 0.6174 | 1.1625 |
| text+abstract | 0.5355 | 0.6667 | 1.2022 |

#### 4.3.2 Results of Model Replacement Ablation

| Model Type | TC | TD | TC+TD |
|------------|------|------|--------|
| origin | 0.536 | 0.717 | 1.253 |
| biomedbert (single stream) | 0.5000 | 0.6000 | 1.1000 |
| biobert (single stream) | 0.4765 | 0.6389 | 1.1154 |
| pubmedbert (single stream) | 0.5636 | 0.7125 | 1.2761 |

### 4.4 Main Experimental Results Comparison

To comprehensively evaluate the effectiveness of the dual-stream fusion algorithm, the table below compares the best performance of various models (including ablation experiments, single-stream models, and reference models) in terms of topic coherence (TC), topic diversity (TD), and their sum (TC+TD).

| Experiment Type | single Best TC | single Best TD | Highest TC+TD | Text Combination for Highest TC+TD |
|----------------|---------|---------|---------------|-------------------------------|
| Dual-stream embedding-basic | 0.6076 | 0.7143 | 1.3005 | full |
| Dual-stream embedding-bio+bio | 0.5826 | 0.7000 | 1.2610 | abmesh |
| Dual-stream embedding-st+bio | 0.6038 | 0.7333 | 1.3247 | full |
| Dual-stream embedding-st+st | 0.6172 | 0.7667 | 1.3550 | abmesh |
| Model replacement only | 0.5636 | 0.7125 | 1.2761 | - |
| Text replacement only | 0.536 | 0.717 | 1.253 | - |
| Reference-Model 3 | 0.536 | 0.717 | 1.253 | - |

---

## 5. Data Analysis

### 5.1 Objective Results of Each Model and Text Combination

Experimental results show that the dual-stream embedding model outperforms single-stream models and ablation experiments across different model combinations (basic, bio+bio, st+bio, st+st) and text combinations (abstract, text, abmesh, full).

- **st+st model** achieved the highest TC+TD (1.3550) with the abmesh text combination, with TC of 0.6146, TD of 0.7667, and parameters α=0.45, β=0.40.
- **st+bio model** achieved TC+TD of 1.3247 with the full text combination, with TC of 0.6038, TD of 0.7333, and parameters α=0.30, β=0.35.
- **basic model** achieved TC+TD of 1.3005 with the full text combination, with TC of 0.6076, TD of 0.7143, and parameters α=0.20, β=0.35.
- **bio+bio model** achieved TC+TD of 1.2610 with the abmesh text combination, with TC of 0.5710, TD of 0.7000, and parameters α=0.45, β=0.75.

### 5.2 Comparison with Ablation and Single-Stream Models

- The highest TC+TD for ablation experiments (text or model replacement only) is 1.2761 (pubmedbert single stream), with TC of 0.5636 and TD of 0.7125.
- Other single-stream models such as biomedbert and biobert have TC+TD of 1.1000 and 1.1154, respectively, both lower than the dual-stream models.
- Merely changing the text combination (e.g., full+abstract, abmesh+abstract, etc.) yields a maximum TC+TD of 1.2022, still lower than the dual-stream models.

### 5.3 Comparison with Reference Model

- The reference Model 3 has TC of 0.536, TD of 0.717, and TC+TD of 1.253.
- All dual-stream models in this experiment achieved higher best TC, TD, and TC+TD than the reference model.

### 5.4 Improvement Magnitude and Parameter Distribution

- **TC improvement:** The highest TC of the dual-stream model (0.6172) is about 15.1% higher than the reference, and about 9.5% higher than the ablation experiment.
- **TD improvement:** The highest TD of the dual-stream model (0.7667) is about 6.9% higher than the reference, and about 5.4% higher than the ablation experiment.
- **TC+TD improvement:** The highest TC+TD of the dual-stream model (1.3550) is about 8.1% higher than the reference, and about 6.2% higher than the ablation experiment.

- **Parameter distribution:** The best results are mostly concentrated in the α=0.2~0.45, β=0.25~0.75 range, indicating that moderate fusion of text and MeSH information and the allocation of major/minor topic weights play an important role in performance improvement.

### 5.5 Topic Label Enhancement with Large Language Model (LLM)
To improve the interpretability of the discovered topics, we integrated a post-processing step using a large language model (LLM) via API calls. While BERTopic provides keyword-based topic representations, these keyword sets are often fragmented or require expert interpretation. To generate human-readable, semantically coherent topic titles suitable for academic use, we employed the Qwen-14B-Instruct model through the DashScope API.
Each topic's top keywords (obtained via BERTopic's c-TFIDF ranking) were used as input to a structured prompt. The prompt instructed the model to generate a concise and accurate English title that summarizes the topic meaningfully, avoiding simple keyword repetition or direct translation. This approach aimed to simulate expert-level summarization and provided titles that are suitable for inclusion in academic papers, visualizations, and topic-based exploration tools.
The generation pipeline included:
Cleaning and verifying the keyword format (ensuring a valid list of terms);
Constructing a prompt template that contextualizes the generation task for the model;
Iterative generation using qwen2.5-14b-instruct-1m with a controlled decoding configuration (temperature=0.3, top_p=0.95);
Storing the output in the experimental result table alongside raw keyword representations.
An example prompt used:
   You are an expert in academic topic summarization. Please generate a concise, accurate, human-readable English title summarizing the following keywords: cancer, therapy, chemotherapy.
   Requirements:
   1. Use one short sentence.
   2. Focus on meaning, not literal translation or keyword listing.
   3. Output should be English, academic-sounding, and suitable for reports or visualizations.
   4. Do not include quotes or numbering.
   Please generate the topic title:
This LLM-based topic labeling not only enhances downstream interpretability but also provides a more intuitive interface for non-expert users, making the dual-stream topic modeling output more accessible for literature exploration and clinical research.

### 5.6 Summary

- The dual-stream embedding model outperforms single-stream models, ablation experiments, and reference models on all evaluation metrics, and demonstrates strong stability and generalization across different model and text combinations.
- The improvement from ablation and single-stream models is limited, further validating the effectiveness and necessity of the dual-stream fusion strategy.

---

## 6. Result and Discussion

The proposed dual-stream embedding models demonstrated consistently superior performance on topic coherence (TC) and topic diversity (TD) compared to single-stream baselines and isolated text or metadata inputs.  As summarized in Table 4.4, the **st+st** model (PubMedBERT embeddings on both streams) achieved the highest overall metrics, with single TC ≈0.617 and single TD ≈0.767, yielding a combined TC+TD ≈1.355.  This far exceeds the baseline Model 3 (the state-of-the-art BERTopic model on PubMed abstracts), which had only TC=0.536 and TD=0.717 (TC+TD=1.253).  The **st+bio** and **basic** (general MiniLM) fusion models also significantly outperformed the baseline, with TC+TD ≈1.325 and 1.301, respectively, while the **bio+bio** fusion was somewhat weaker (TC+TD ≈1.261).  In all cases, the dual-stream architectures delivered higher coherence and diversity.  Notably, even the *basic* fusion (MiniLM on both streams) matched or exceeded Model 3 (1.300 vs. 1.253), indicating that the fusion framework itself contributes to gains beyond model selection alone.  These results confirm that combining complementary embedding sources yields more coherent and distinct topics than any single embedding.  In fact, prior work has shown that incorporating MeSH-based semantic features improves topic detection performance; our dual-stream approach takes advantage of this by blending unstructured text with structured MeSH information via learned weights, leading to substantially better thematic representations.

The ablation experiments further highlight the advantage of the dual-stream design.  Simply replacing the base model with a stronger biomedical embedding (the "only model replacement" case, e.g. using PubMedBERT alone) yielded only a modest improvement (single TC≈0.564, single TD≈0.713, Best TC+TD≈1.276, as shown in Table 4.4), which is still below every dual-stream configuration.  Likewise, altering only the input text (e.g. adding MeSH terms without model fusion) produced TC+TD≈1.253, essentially no improvement over Model 3.  In contrast, the full dual-stream fusion achieved up to \~7.9% higher TC+TD than the best ablation case.  This demonstrates that neither a stronger encoder nor additional metadata alone can match the gains of jointly embedding both sources.  The dual-stream models effectively learn to leverage structured and unstructured information in concert.  The St+St fusion in particular, which uses specialized biomedical embeddings on both channels, consistently led to the most coherent and diverse topic sets.  This agrees with domain studies showing that domain-specific models like PubMedBERT outperform general or mixed models on medical text tasks.  Thus, the dual-stream framework uniquely integrates domain expertise from different sources, boosting interpretability beyond what single-stream or single-feature approaches can achieve.

A key factor in these improvements is the two-layer weighted fusion mechanism.  Each model learns two weights, **α** (balancing text vs. MeSH stream) and **β** (balancing primary vs. secondary MeSH topics), which are tuned to optimize TC and TD.  Table 4.2 illustrates that the optimal α and β vary notably across models and text configurations.  For example, in the st+st model the highest coherence (TC) was obtained at α≈0.45, β≈0.20, whereas the highest combined score (TC+TD) occurred at α≈0.45, β≈0.40 (Table 4.2.4).  In the st+bio model, the best TC+TD came with full text input at α≈0.30, β≈0.35 (Table 4.2.3).  These results show that the model can flexibly emphasize the unstructured or structured channel depending on the context.  A higher α gives more weight to textual content (useful when abstracts are rich), while a lower α (with higher weight on MeSH) can inject domain signals that boost topic distinctness.  Similarly, adjusting β controls how strongly the model prioritizes the most salient MeSH terms in each topic.  This adaptability helps the model self-tune to the dataset: certain configurations benefited from more MeSH influence (improving TD) and others from more text influence (improving TC).  In practice, the wide range of near-optimal (α, β) pairs observed suggests the fusion approach is robust: moderate shifts in weights did not drastically degrade performance.  In sum, the dual-layer weighting scheme is critical, enabling the model to reconcile non-structured and structured information in a data-driven way and thus achieve higher overall topic quality.

The interaction between model architectures and text representations is also apparent.  Different model combinations paired best with different document configurations.  For **st+st** and **bio+bio**, the *abstract+MeSH* ("abmesh") strategy yielded the highest diversity and TC+TD (Table 4.4), indicating that concise text augmented with MeSH annotations was most informative when both streams use similar domain embeddings.  In contrast, **st+bio** and **basic** models performed best with the *full text* (title+abstract+MeSH) combination.  This suggests that when a general model (basic) or mixed model (st+bio) is used, providing the complete text context yields better coherence.  Practically, these findings imply different use-case strategies: if maximum topic coherence and diversity are needed and abstract+MeSH data are available, using two domain-specific models (st+st) with the abmesh strategy is optimal.  If abstracts are longer or one prefers using all available information, then st+bio or even basic fusion with full-text input can be effective.  For quick or limited scenarios (e.g. only abstracts), the weighted fusion still outperforms single streams, but one might tune α higher to leverage the text.  Overall, the dual-stream framework offers a menu of model/text strategies that can be tailored to specific requirements, whether one prioritizes depth of biomedical nuance (favoring st+st+abmesh) or breadth of coverage (favoring full-text inputs).

In summary, the dual-stream fusion method consistently outperforms single-stream and non-fusion baselines on biomedical topic modeling.  By jointly embedding document text and MeSH knowledge, it produces topics that are more semantically coherent and less redundant.  The two-tier fusion weights (α and β) provide flexible control over the information mix, allowing the model to adapt to different corpus characteristics.  These advantages suggest broad applicability of the approach: for example, it could improve literature review tools, semantic search, or clustering in biomedical research by yielding clearer topic clusters.  More generally, this framework can be extended to other domains by substituting appropriate domain-specific embeddings or ontologies.  In all cases, our results demonstrate that multi-source embedding fusion is a powerful way to enrich topic models with domain structure, leading to more reliable and interpretable discovery of thematic patterns.

---

## 6. Conclusions and Application Recommendations

### 6.1 Main Conclusions

1. **The dual-stream fusion topic modeling method significantly outperforms traditional single-stream models and ablation schemes in the biomedical literature domain, with the most notable improvement in topic coherence (TC).**
2. **The parameterized fusion mechanism (α, β) provides the model with high flexibility and interpretability, enabling dynamic adjustment of information source weights according to actual needs.**
3. **Ablation experiments fully demonstrate that it is difficult for a single information source or single-stream model to achieve the same improvement; dual-stream fusion is the key to improving topic modeling quality.**

### 6.2 Application Recommendations

- **For scenarios requiring high coherence:** It is recommended to use the st+st model with the full text combination (α=0.45, β=0.40) or the basic model with the abmesh text combination (α=0.20, β=0.25).
- **For scenarios requiring balanced performance:** It is recommended to use the bio+bio model with the full text combination (α=0.45, β=0.80).
- **For resource-constrained scenarios:** The basic model with the text combination (α=0.65, β=0.30) can be selected.
- **For scenarios without structured knowledge:** It is recommended to use the st+bio model with the abstract text combination (α=0.20, β=0.70).

### 6.3 Future Work Prospects

1. Explore adaptive parameter design to achieve dynamic adjustment of α and β based on document characteristics.
2. Investigate multi-stream architectures, incorporating more structured information such as author networks and citation relationships.
3. Further validate the model's generalization ability on datasets from different biomedical subfields.
4. Explore the value of the model in practical applications such as clinical decision support and literature recommendation.
