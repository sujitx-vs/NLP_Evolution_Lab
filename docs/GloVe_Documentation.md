# GloVe (Global Vectors for Word Representation)

## Introduction

GloVe (Global Vectors for Word Representation) is a word embedding model
introduced by researchers at Stanford University in 2014. It was
designed to learn meaningful vector representations of words by
combining the advantages of global corpus statistics with neural
embedding techniques.

------------------------------------------------------------------------

# Definition

GloVe is an **unsupervised word embedding model** that learns dense
vector representations of words by analyzing **global word co-occurrence
statistics** across an entire text corpus.

Unlike Word2Vec, which learns by predicting neighboring words, GloVe
learns from how frequently words appear together throughout the corpus.

------------------------------------------------------------------------

# Why was GloVe introduced?

Word2Vec was a major breakthrough, but it had some limitations.

Problems with Word2Vec:

-   Learns mainly from a small local context (sliding window).
-   Does not directly use global corpus statistics.
-   Some semantic relationships are not captured as effectively.

Researchers wanted a model that could use information from the **entire
corpus** rather than only nearby words.

This led to the development of GloVe.

------------------------------------------------------------------------

# Problem GloVe Solves

GloVe solves the problem of learning better semantic word embeddings by
utilizing **global co-occurrence information**.

Instead of asking:

> "Can I predict the next word?"

it asks:

> "How often do these two words appear together in the entire corpus?"

This allows the model to capture richer semantic relationships.

------------------------------------------------------------------------

# Core Idea

The fundamental idea behind GloVe is:

> Words that frequently occur together should have similar vector
> representations.

For example:

    ice     → cold
    snow    → cold
    water   → liquid

By studying word co-occurrence frequencies across millions of sentences,
GloVe learns embeddings that preserve these relationships.

------------------------------------------------------------------------

# Architecture

Unlike Word2Vec, GloVe does **not** train using CBOW or Skip-Gram.

Instead, it follows this process:

    Text Corpus
          ↓
    Build Vocabulary
          ↓
    Create Co-occurrence Matrix
          ↓
    Optimize Word Vectors
          ↓
    Learn Embeddings

The model learns vectors by minimizing the difference between predicted
and actual co-occurrence relationships.

------------------------------------------------------------------------

# How GloVe Works

1.  Collect a large text corpus.
2.  Build the vocabulary.
3.  Count how often every pair of words appears together.
4.  Create a Word Co-occurrence Matrix.
5.  Initialize word vectors randomly.
6.  Optimize the vectors using the co-occurrence matrix.
7.  Update vectors through gradient descent.
8.  Save the learned embeddings.

Unlike Word2Vec, there is **no prediction of missing words** during
training.

------------------------------------------------------------------------

# Workflow

    Large Text Corpus
            ↓
    Tokenization
            ↓
    Vocabulary
            ↓
    Co-occurrence Matrix
            ↓
    Optimization
            ↓
    Word Embeddings

------------------------------------------------------------------------

# Output

After training, every word receives a dense embedding vector.

Example:

    computer

    ↓

    [-0.34, 0.71, -0.12, ...]

These embeddings can be used as input to other NLP models such as RNNs,
LSTMs, Transformers, and BERT.

------------------------------------------------------------------------

# Applications

-   Sentiment Analysis
-   Text Classification
-   Machine Translation
-   Question Answering
-   Information Retrieval
-   Named Entity Recognition

------------------------------------------------------------------------

# Advantages

-   Uses global corpus statistics.
-   Produces high-quality semantic embeddings.
-   Captures many word relationships effectively.
-   Efficient once the co-occurrence matrix is built.
-   Widely used as pretrained embeddings.

------------------------------------------------------------------------

# Limitations

## 1. Static Embeddings

Every word has only one vector regardless of context.

Example:

    bank (river)

    bank (financial)

Both receive the same embedding.

------------------------------------------------------------------------

## 2. Large Memory Requirement

Building the co-occurrence matrix can require significant memory for
very large vocabularies.

------------------------------------------------------------------------

## 3. Out-of-Vocabulary Words

Words not seen during training cannot be represented.

------------------------------------------------------------------------

## 4. No Context Awareness

The embedding does not change depending on the sentence.

------------------------------------------------------------------------

# Comparison with Word2Vec

  ------------------------------------------------------------------------
  Feature                  Word2Vec                    GloVe
  ------------------------ --------------------------- -------------------
  Training Method          Predict nearby words        Learn from
                                                       co-occurrence
                                                       matrix

  Information Used         Local context               Global corpus
                                                       statistics

  Embedding Type           Static                      Static

  Neural Network           Shallow Neural Network      Matrix
                                                       factorization +
                                                       optimization

  Output                   Word Embeddings             Word Embeddings
  ------------------------------------------------------------------------

------------------------------------------------------------------------

# Historical Significance

    One-Hot Encoding
            ↓
    Word2Vec
            ↓
    GloVe
            ↓
    RNN

GloVe improved upon Word2Vec by incorporating global statistical
information from the entire corpus. It became one of the most widely
used pretrained embedding models before the rise of contextual
embeddings such as ELMo, BERT, and GPT.

------------------------------------------------------------------------

# Key Takeaways

-   GloVe stands for Global Vectors for Word Representation.
-   Learns embeddings using global word co-occurrence statistics.
-   Produces dense semantic word vectors.
-   Does not predict missing words like Word2Vec.
-   Uses the entire corpus to learn relationships.
-   Still produces static embeddings.
-   Its limitations motivated the development of contextual embedding
    models such as BERT.
