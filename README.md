# NLP Evolution Lab

A personal playground for exploring and understanding the evolution of Natural Language Processing (NLP).

This repository is **not** a production-ready project. It is a collection of experiments, notes, comparisons, and implementations created while learning how NLP has evolved over the years — from simple rule-based systems to modern LLM-powered applications.

The goal is not only to learn **how** different NLP techniques work, but also **why** they were introduced, what limitations they solved, and how the representation of language became richer over time.

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

## Understanding the Evolution

One important idea while studying NLP is that newer models should not always be interpreted as simply being **better versions of older models**.

From machine-learning-based NLP onward, different models and architectures often solve different problems or introduce different capabilities.

For example:

- Word2Vec and GloVe improve **word representation**.
- RNN, LSTM, and GRU improve **sequence modeling**.
- Seq2Seq introduces an **encoder-decoder architecture** for sequence transformation.
- Attention improves access to relevant information across a sequence.
- Transformers redesign sequence modeling around attention.
- Pretrained Transformer models learn rich contextual representations that can be adapted to many downstream tasks.

Therefore, the evolution of NLP should be viewed as an expansion of capabilities rather than a simple ranking where every new model completely replaces the previous one.

A useful conceptual progression is:

```text
TEXT REPRESENTATION

One-hot / BoW / TF-IDF
        ↓
Word2Vec / GloVe
        ↓


SEQUENCE MODELING

Vanilla RNN
        ↓
LSTM / GRU
        ↓


SEQUENCE-TO-SEQUENCE ARCHITECTURE

Encoder + Decoder
        ↓


ATTENTION MECHANISM

Decoder can inspect encoder states
        ↓


TRANSFORMER ARCHITECTURE

Attention becomes central
        ↓


PRETRAINED TRANSFORMER MODELS

BERT / GPT / T5 / etc.
        ↓


LARGE LANGUAGE MODELS
```

This distinction is important because these are not always models at the same conceptual level.

For example:

```text
RNN / LSTM / GRU
= recurrent sequence-processing architectures

Seq2Seq
= an encoder-decoder architecture that can use RNN, LSTM, or GRU

Attention
= a mechanism for selecting relevant information

Transformer
= an architecture built primarily around attention
```

So the NLP journey is not simply:

```text
Model A
↓
Model B
↓
Model C
```

It is better understood as an evolving toolbox of representations, architectures, and learning mechanisms.

---

## Representations Became Richer

Another important way to understand NLP evolution is this:

> The journey is not only about models becoming more capable.

It is also about **language representations becoming richer**.

### Bag of Words / TF-IDF

```text
sentence
↓
sparse numerical counts or weights
```

These approaches mainly capture:

- which words appear
- how frequently they appear
- how important a word may be within a document

They provide little or no semantic understanding of relationships between words.

---

### Word2Vec / GloVe

```text
word
↓
dense semantic vector
```

Words are represented using dense vectors where words with similar usage patterns can appear close to each other in the embedding space.

For example:

```text
king
queen
prince
```

may have more similar vectors than:

```text
king
banana
```

This was a major improvement over sparse count-based representations.

However, these embeddings are **static**.

The word:

```text
bank
```

has the same vector in:

```text
river bank
```

and:

```text
bank account
```

---

### RNN

```text
token at position t
↓
representation influenced by previous tokens
```

RNNs introduce sequential context.

Instead of considering a word completely independently, the hidden state contains information from previous positions in the sequence.

Conceptually:

```text
current token
+
previous context
↓
new hidden representation
```

---

### LSTM / GRU

```text
token at position t
↓
contextual representation
with improved long-term memory
```

LSTM and GRU improve the ability of recurrent networks to preserve important information across longer sequences.

They introduce gating mechanisms that control what information should be:

```text
kept
forgotten
updated
exposed
```

This helps reduce the long-term dependency problems found in vanilla RNNs.

---

### Transformer

```text
token at position t
↓
representation influenced by relationships
with many or all other relevant tokens
```

Transformers move away from recurrence and use attention mechanisms to model relationships between tokens.

A token representation can therefore depend on information from many different positions in the sequence.

This enables much richer contextual representations.

---

## A Simple Representation Timeline

```text
BoW / TF-IDF
=
"What words are present?"


Word2Vec / GloVe
=
"What words have similar semantic usage?"


RNN
=
"What has happened earlier in the sequence?"


LSTM / GRU
=
"What earlier information should be preserved?"


Seq2Seq
=
"How can one sequence be transformed into another?"


Attention
=
"Which parts of the sequence are most relevant right now?"


Transformer
=
"How do tokens relate to other tokens across the sequence?"
```

This is one of the core perspectives used throughout this repository.

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
Need semantic word representations
        ↓
Word Embeddings
        ↓
Need sequence and contextual information
        ↓
RNN / LSTM / GRU
        ↓
Need sequence-to-sequence transformation
        ↓
Seq2Seq
        ↓
Need better access to long-range information
        ↓
Attention
        ↓
Need scalable contextual modeling
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
- [x] Word Tokenization
- [x] Stopword Removal
- [x] Stemming
- [x] Lemmatization
- [x] POS Tagging
- [x] Named Entity Recognition
- [x] Bag of Words
- [x] TF-IDF
- [x] N-grams
- [x] Hidden Markov Models
- [x] SVM Text Classification
- [x] Word2Vec
- [x] GloVe
- [x] RNN
- [x] LSTM
- [x] GRU
- [ ] Seq2Seq
- [ ] Attention
- [ ] Transformer
- [ ] ...

---

## Repository Structure

```text
datasets/
    Sample datasets used during learning

src/
    Python experiments and implementations

notes/
    Observations, comparisons, and learning notes

models/
    Saved experimental models and embeddings
```

---

## Note

This repository intentionally contains experimental code, failed attempts, comparisons, and rough implementations.

The focus is on understanding the **evolution of ideas in NLP**, including why different techniques were introduced, what problems they solved, and what limitations motivated the next stage.

Think of this repository as a learning journal rather than a polished software project.