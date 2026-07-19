# NLP Evolution Lab

A personal playground for exploring and understanding the evolution of Natural Language Processing (NLP).

This repository is **not** a production-ready project. It is a collection of experiments, notes, comparisons, and implementations created while learning how NLP has evolved over the years—from simple rule-based systems to modern LLM-powered applications.

The goal is not only to learn **how** different NLP techniques work, but also **why** they were introduced and what limitations they solved.

---

## Learning Roadmap

```text
1980s - Rule-Based NLP
        │
        ├── Text Processing
        ├── Regex
        ├── Sentence Segmentation
        ├── Tokenization
        ├── Stopword Removal
        ├── Stemming
        ├── Lemmatization
        ├── POS Tagging
        ├── Named Entity Recognition
        └── Rule-Based Parsing

1990s - Statistical NLP
        │
        ├── Bag of Words
        ├── TF-IDF
        ├── N-grams
        ├── Hidden Markov Models (HMM)
        ├── Naive Bayes
        ├── SVM
        └── Statistical Parsing

2010s - Deep Learning
        │
        ├── Word2Vec
        ├── GloVe
        ├── FastText
        ├── RNN
        ├── LSTM
        ├── GRU
        ├── Seq2Seq
        └── Attention

2017+ - Transformer Era
        │
        ├── Transformer
        ├── BERT
        ├── GPT
        ├── RoBERTa
        └── Sentence Transformers

Modern NLP
        │
        ├── Embeddings
        ├── Vector Databases
        ├── Semantic Search
        ├── Hybrid Search
        ├── RAG
        ├── AI Agents
        ├── Tool Calling
        └── Agent Workflows
```

---

## Repository Goal

The objective of this repository is to understand the complete NLP workflow through one evolving project.

Instead of learning isolated topics, every new concept is introduced because the previous approach reaches its limitations.

For example:

```text
Naive Sentence Splitter
        ↓
Fails on abbreviations and decimals
        ↓
Sentence Tokenizer (NLTK / spaCy)
        ↓
Need better language understanding
        ↓
Statistical NLP
        ↓
Need contextual understanding
        ↓
Transformers
        ↓
Need external knowledge
        ↓
RAG
        ↓
Need planning and actions
        ↓
AI Agents
```

---

## Current Progress

- [x] Reading Text Files
- [x] Text Normalization
- [x] Regex Basics
- [x] Naive Sentence Segmentation
- [x] NLTK Sentence Tokenization
- [ ] Word Tokenization
- [ ] Stopword Removal
- [ ] Stemming
- [ ] Lemmatization
- [ ] POS Tagging
- [ ] Named Entity Recognition
- [ ] ...

---

## Repository Structure

```
datasets/
    Sample datasets used during learning

src/
    Python experiments and implementations

notes/
    Observations, comparisons, and learning notes
```

---

## Note

This repository intentionally contains experimental code, failed attempts, comparisons, and rough implementations. Think of it as a learning journal rather than a polished software project.