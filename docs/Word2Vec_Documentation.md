# Word2Vec

## Introduction

Word2Vec is one of the first successful deep learning models for
learning meaningful numerical representations of words. Introduced by
Google in 2013, it transformed NLP by replacing sparse one-hot vectors
with dense semantic embeddings.

------------------------------------------------------------------------

# Definition

Word2Vec is a **shallow neural network** that learns vector
representations (embeddings) of words using a self-supervised learning
task on a large text corpus.

Instead of manually defining word meanings, Word2Vec learns them
automatically from how words appear in context.

------------------------------------------------------------------------

# Why was Word2Vec introduced?

Before Word2Vec, words were represented using **One-Hot Encoding**.

Example:

    cat  -> [0 1 0 0 0]
    dog  -> [0 0 1 0 0]
    milk -> [0 0 0 1 0]

Problems with One-Hot Encoding:

-   Very high-dimensional and sparse vectors.
-   No semantic meaning.
-   Every pair of words is equally distant.
-   Impossible to understand that "cat" and "dog" are more similar than
    "cat" and "car".

Researchers wanted a representation where words with similar meanings
had similar numerical representations.

This led to Word2Vec.

------------------------------------------------------------------------

# Problem Word2Vec Solves

Word2Vec solves the problem of **learning semantic word
representations**.

Instead of representing a word as an index, it represents a word as a
dense vector that captures semantic relationships.

Example:

    king ≈ queen

    cat ≈ dog

    Paris ≈ France

------------------------------------------------------------------------

# Core Idea

The idea is simple:

> If two words appear in similar contexts, they probably have similar
> meanings.

Example:

    The cat drinks milk.
    The dog drinks milk.

Since "cat" and "dog" appear in similar contexts, Word2Vec learns
similar embeddings for them.

------------------------------------------------------------------------

# Architectures

Word2Vec provides two training strategies.

## 1. Continuous Bag of Words (CBOW)

Predict the target word from its surrounding context.

Example:

    The ____ drinks milk

    ↓

    cat

Context → Target Word

------------------------------------------------------------------------

## 2. Skip-Gram

Predict surrounding words using the current word.

Example:

    Input:

    cat

    ↓

    Predict:

    The
    drinks
    milk

Target Word → Context

------------------------------------------------------------------------

# How Word2Vec Works

1.  Collect a large text corpus.
2.  Build the vocabulary.
3.  Generate training pairs using CBOW or Skip-Gram.
4.  Convert input words into one-hot vectors.
5.  Pass the one-hot vector through the embedding matrix (W).
6.  Obtain the word embedding.
7.  Predict the target/context word using the output matrix (W').
8.  Compute Cross-Entropy Loss.
9.  Update W and W' using Backpropagation and Gradient Descent.
10. After training, discard W' and keep W as the learned embedding
    matrix.

------------------------------------------------------------------------

# Workflow

    Text Corpus
          ↓
    Tokenization
          ↓
    Training Pairs
          ↓
    One-Hot Encoding
          ↓
    Embedding Matrix (W)
          ↓
    Word Embedding
          ↓
    Output Layer (W')
          ↓
    Prediction
          ↓
    Loss
          ↓
    Backpropagation
          ↓
    Updated Embeddings

------------------------------------------------------------------------

# Output

After training, every word has a dense vector.

Example:

    cat

    ↓

    [0.42, -0.18, 0.71, ...]

These embeddings can then be used in other NLP models.

------------------------------------------------------------------------

# Applications

-   Sentiment Analysis
-   Machine Translation
-   Text Classification
-   Named Entity Recognition
-   Question Answering
-   Information Retrieval

------------------------------------------------------------------------

# Advantages

-   Learns semantic relationships.
-   Dense, low-dimensional vectors.
-   Fast to train.
-   Supports similarity and analogy tasks.
-   Foundation for modern word embeddings.

------------------------------------------------------------------------

# Limitations

## 1. Static Embeddings

A word always has the same vector regardless of context.

Example:

    bank (river)

    bank (financial institution)

Both receive the same embedding.

------------------------------------------------------------------------

## 2. Local Context

Word2Vec only learns from a small sliding window and ignores the broader
document context.

------------------------------------------------------------------------

## 3. Out-of-Vocabulary Words

Words not seen during training cannot receive embeddings.

------------------------------------------------------------------------

## 4. Requires Large Training Data

High-quality embeddings require a large text corpus.

------------------------------------------------------------------------

# Historical Significance

    One-Hot Encoding
            ↓
    Word2Vec
            ↓
    GloVe
            ↓
    RNN

Word2Vec introduced dense semantic embeddings and changed how words were
represented in NLP. It became the foundation for many later deep
learning models.

------------------------------------------------------------------------

# Key Takeaways

-   Word2Vec is a shallow neural network.
-   It learns word embeddings using self-supervised learning.
-   Two training methods: CBOW and Skip-Gram.
-   The learned embedding matrix is the final product.
-   Similar words obtain similar vectors.
-   Its limitations motivated improved embedding methods such as GloVe
    and later contextual embeddings like BERT.
