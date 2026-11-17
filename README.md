# 🧠 NLP Project — Question Difficulty Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

**A complete NLP pipeline for classifying question difficulty (Easy / Medium / Hard) using classical ML, deep learning, and transformer-based models.**

</div>

---

## 📌 Project Overview

This project addresses a real-world **Natural Language Processing** task: given a trivia question, automatically predict its **difficulty level** (Easy, Medium, or Hard). The dataset is derived from a large-scale QA corpus (`train.xls`) containing over 90,000 labeled questions.

The project is structured as a progressive learning pipeline across **6 phases**, starting from raw data preprocessing up to fine-tuning large pre-trained language models (BERT).

---

## 🗂️ File Structure

```
NLP-project/
│
├── ex1.ipynb                                          # Phase 1 — Text Preprocessing Pipeline
├── nlp project.ipynb                                  # Phase 2 — Feature Engineering (TF-IDF, Word2Vec)
├── PART3_CLASSIFFICATION.ipynb                        # Phase 3 — Classical ML Classification
├── PART4_CLUSTERING.ipynb                             # Phase 4 — Unsupervised Clustering & Relation Extraction
├── PART5_Deep Learning for Text Classification.ipynb  # Phase 5 — Deep Learning (RNN)
├── PART6_Transformers.ipynb                           # Phase 6 — Fine-tuning BERT (bert-base/large-cased)
└── דוח עבודה חלק 1.docx                              # Written report for Phase 1 (Hebrew)
```

---

## 📂 Notebook Descriptions

### 📘 `ex1.ipynb` — Text Preprocessing Pipeline
**Role:** Foundation of the entire project — raw data ingestion and text normalization.

- Loads the dataset (`train.xls`, 90,425 rows) with 6 text columns: `question`, `context`, `answer`, `level`, `type`, `quotes`
- Performs intelligent auto-detection of text columns
- Applies a full NLP preprocessing pipeline per column:
  - URL/Email/handle removal
  - NLTK word tokenization
  - POS tagging (averaged perceptron tagger)
  - **Lemmatization** with WordNet (POS-aware mapping)
  - Proper noun filtering (NNP/NNPS skipped)
  - `be`-form normalization (am/is/are → *be*)
  - Digit replacement with `_number`
  - Stopword removal (NLTK English list)
- Outputs both a fully preprocessed dataframe and a partially preprocessed one for comparison

---

### 📘 `nlp project.ipynb` — Feature Engineering
**Role:** Transform clean text into numerical feature vectors ready for ML models.

- Builds **TF-IDF** feature matrices (10,000 max features, unigrams)
- Trains a **Word2Vec** model (skip-gram, 100-dim, window=5) on training tokens
- Creates document-level vectors from Word2Vec using **TF-IDF–weighted averaging**

---

### 📘 `PART3_CLASSIFFICATION.ipynb` — Classical ML Classification
**Role:** Systematic benchmarking of classical ML classifiers in both binary and multi-class settings.

#### Data Setup
- Dataset: `train-filtered_question_level.csv` — 90,418 deduplicated question-level pairs
- Stratified split: **70% train / 15% val / 15% test** (preserving class ratios)
- Text preprocessing: same NLTK pipeline (tokenization + POS lemmatization)
- Feature representations: TF-IDF (10K features) + Word2Vec (100-dim, TF-IDF weighted)

#### Binary Classification (Easy vs. Hard)
| Model | Representation | Accuracy | F1 Score |
|---|---|---|---|
| Naive Bayes (Multinomial) | TF-IDF | 63.65% | 0.553 |
| **Logistic Regression** | **TF-IDF** | **70.83%** | **0.686** |
| Naive Bayes (Gaussian) | Word2Vec | 58.48% | 0.477 |
| Logistic Regression | Word2Vec | 66.03% | 0.605 |

#### Multi-class Classification (Easy / Medium / Hard)
| Model | Representation | Accuracy | Macro F1 |
|---|---|---|---|
| Naive Bayes (Multinomial) | TF-IDF | 63.45% | 0.286 |
| **Logistic Regression** | **TF-IDF** | **66.56%** | **0.432** |
| Naive Bayes (Gaussian) | Word2Vec | 46.99% | 0.386 |
| Logistic Regression | Word2Vec | 63.49% | 0.323 |

> **Key Finding:** TF-IDF consistently outperforms Word2Vec across all models and tasks. Word2Vec was discontinued after Part 3 in favor of TF-IDF exclusively.

#### Hyperparameter Tuning (Naive Bayes alpha sweep)
| alpha | Accuracy | Macro F1 |
|---|---|---|
| 0.01 | 63.25% | **0.309** |
| 0.1 | 63.30% | 0.307 |
| 1.0 | 63.45% | 0.286 |
| 2.0 | 63.28% | 0.275 |

---

### 📘 `PART4_CLUSTERING.ipynb` — Unsupervised Clustering & Relation Extraction
**Role:** Explore the latent structure of questions without labels, then extract linguistic knowledge.

#### Part A — Clustering
- **TF-IDF vectorization** (90,418 questions, 10K features)
- **TruncatedSVD** dimensionality reduction: 10,000 → 50 dimensions
- **DBSCAN** on a 20,000-sample subset:
  - 20 clusters discovered + noise class
  - Noise ratio: ~70% (expected for short-text trivia)
- **t-SNE** visualization (5,000-point subsample)
- **Cluster evaluation:**

| Metric | Value |
|---|---|
| Purity Score | **0.6312** |
| Adjusted Rand Index (ARI) | -0.0069 |

#### Cluster Topic Labels (via TF-IDF keyword analysis)
| Cluster | Topic |
|---|---|
| 0 | Historical Chronology & Governance |
| 1 | Dated Biographies & Global Events |
| 2 | Music Bands & Rock Culture |
| 3 | Authorship & Literary Works |
| 4 | Geographic Landmarks & Districts |

#### Part B — Relation Extraction
- **spaCy** (`en_core_web_sm` / `en_core_web_lg`) for dependency parsing
- **textacy** for Subject-Verb-Object (SVO) triple extraction
- Custom matchers for `Attribute` and `Location` relation types
- **Refined extraction** to filter noise (pronoun subjects, fragmented dates) → 9 high-quality triples

---

### 📘 `PART5_Deep Learning for Text Classification.ipynb` — RNN-based Classification
**Role:** Apply deep learning to the classification task; compare embedding strategies.

- **Dataset**: balanced 7,000 samples/class (21,000 total); stratified 70/15/15 split
- **Tokenizer**: Keras `Tokenizer` with `<OOV>` token; vocabulary size: 36,543
- **Sequence padding**: fixed MAX_LEN determined by 95th-percentile analysis
- **PyTorch DataLoaders** with `TensorDataset`

#### Models Tested

| Experiment | Embedding | Frozen | Dropout | LR | Accuracy (Val) |
|---|---|---|---|---|---|
| Scratch | Random init | — | 0.0 | 0.001 | 49.68% |
| W2V Frozen | Word2Vec | ✅ | 0.0 | 0.001 | 50.99% |
| W2V Fine-tuned | Word2Vec | ❌ | 0.0 | 0.001 | 49.65% |
| RNN + W2V | Word2Vec | ✅ | 0.0 | 0.001 | 41.92% |
| RNN + W2V | Word2Vec | ✅ | 0.2 | 0.0005 | 46.37% |
| RNN + W2V | Word2Vec | ❌ | 0.3 | 0.0005 | **46.47%** |

> **Architecture**: Embedding → Dropout → Linear(128) → ReLU → Dropout → Linear(3)  
> Early stopping with patience=3 to prevent overfitting.

---

### 📘 `PART6_Transformers.ipynb` — BERT Fine-Tuning
**Role:** Leverage pre-trained transformer models for state-of-the-art classification performance.

#### Setup
- **Dataset**: balanced 7,000/class (21,000 total); exact 70/15/15 split
- **Tokenizer**: HuggingFace `AutoTokenizer` (model-native, no manual lowercasing)
- **MAX_LEN**: 44 tokens (95th percentile of question lengths)
- **Device**: CUDA (Colab GPU)
- **Optimizer**: AdamW with warmup (10% of steps) + weight decay

#### Experiments (bert-base-cased)
| Run Name | lr | Batch | Epochs | Val Acc (last epoch) |
|---|---|---|---|---|
| BERT_BASE_lr3e-5_bs16_ep3 | 3e-5 | 16 | 3 | ~56% |
| BERT_BASE_lr2e-5_bs16_ep3 | 2e-5 | 16 | 3 | **57.34%** |
| BERT_BASE_lr3e-5_bs16_ep2 | 3e-5 | 16 | 2 | 57.18% |

#### Experiments (bert-large-cased)
| Run Name | lr | Batch | Epochs | Val Acc (last epoch) |
|---|---|---|---|---|
| BERT_LARGE_lr2e-5_bs8_ep2 | 2e-5 | 8 | 2 | 56.65% |
| BERT_LARGE_lr2e-5_bs8_ep3 | 2e-5 | 8 | 3 | 56.72% |

> **Best overall result**: `BERT-base-cased`, lr=2e-5, batch=16, 3 epochs → **Val Acc ≈ 57.3%**

---

## 📊 Results Summary

| Phase | Model | Setting | Best Accuracy | Best Macro F1 |
|---|---|---|---|---|
| Part 3 | Logistic Regression | Binary (TF-IDF) | **70.83%** | 0.686 |
| Part 3 | Logistic Regression | Multi-class (TF-IDF) | **66.56%** | 0.432 |
| Part 4 | DBSCAN | Clustering | Purity: 0.63 | ARI: -0.007 |
| Part 5 | RNN | W2V Frozen | ~50% | — |
| Part 6 | BERT-base-cased | Fine-tuned | **~57%** | — |

> **Note on difficulty:** This is an inherently hard NLP task — question *difficulty* is subjective and highly context-dependent. A macro F1 of 0.43 on 3 imbalanced classes with classical models, and 57% accuracy with BERT on a balanced set, are strong results for this domain.

---

## 🔧 Installation & Setup

### Prerequisites
```bash
Python >= 3.10
CUDA-capable GPU (recommended for Parts 5 & 6)
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn gensim nltk spacy textacy
pip install torch torchvision torchaudio  # or visit pytorch.org for CUDA-specific install
pip install transformers accelerate datasets
python -m spacy download en_core_web_sm
```

### NLTK Resources (auto-downloaded on first run)
```python
import nltk
nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng',
               'wordnet', 'omw-1.4', 'stopwords'])
```

---

## 🚀 Usage

All notebooks are designed to run in **Google Colab** with the dataset mounted or uploaded manually.

### Step-by-step
1. Upload `train.xls` (or `train-filtered_question_level.csv`) to your Colab session
2. Run notebooks **in order**: `ex1` → `nlp project` → `PART3` → `PART4` → `PART5` → `PART6`
3. Each notebook documents its own preprocessing steps and can be run independently if the prerequisite CSV is available

### Quick Start (Parts 3–6 only)
If you already have `train-filtered_question_level.csv`:
```python
# In any Part 3–6 notebook, update the path:
filtered_df = pd.read_csv("/content/train-filtered_question_level.csv")
```

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Language** | Python 3.12 |
| **Data** | pandas, numpy |
| **NLP Preprocessing** | NLTK (tokenization, POS, lemmatization, stopwords) |
| **Feature Engineering** | scikit-learn TF-IDF, Gensim Word2Vec |
| **Classical ML** | scikit-learn (Naive Bayes, Logistic Regression) |
| **Clustering** | scikit-learn (KMeans, DBSCAN, TruncatedSVD, t-SNE) |
| **Relation Extraction** | spaCy, textacy |
| **Deep Learning** | PyTorch, Keras/TensorFlow (tokenizer) |
| **Transformers** | HuggingFace `transformers` (BERT-base/large-cased) |
| **Visualization** | matplotlib, seaborn |

---

## 👥 Authors

> Project developed as part of an academic NLP course.  
> Dataset: SQuAD-style QA corpus (`train.xls`) with question difficulty labels.

---

## 📄 License

This project is for academic and educational purposes.
