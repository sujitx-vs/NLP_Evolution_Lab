# Recurrent Neural Network (RNN)

## Introduction

A Recurrent Neural Network (RNN) is a neural network architecture
designed to process **sequential data**. Unlike a traditional
feed-forward neural network, an RNN remembers information from previous
inputs while processing the current input.

Common sequential data: - Text - Speech - Time series - DNA sequences

------------------------------------------------------------------------

# Definition

An RNN is a neural network that processes one element of a sequence at a
time while maintaining a **Hidden State**, which acts as its memory.

------------------------------------------------------------------------

# Why was RNN introduced?

Word2Vec and GloVe solved the problem of representing individual words
as dense vectors.

However, they could **not understand sequences**.

Example:

    Dog bites man.

    Man bites dog.

Both sentences contain almost the same words, but their meanings are
different because the **order** of the words is different.

Traditional neural networks also assume that every input is independent.

They cannot naturally remember previous words.

Researchers needed a model that could process words **in order** and
remember what it had already seen.

This led to the Recurrent Neural Network.

------------------------------------------------------------------------

# Problem RNN Solves

RNN solves **sequence modeling**.

Instead of treating every word independently, it processes words one by
one while carrying information from previous words.

Example tasks:

-   Sentiment Analysis
-   Language Modeling
-   Part-of-Speech Tagging
-   Named Entity Recognition
-   Next Word Prediction

------------------------------------------------------------------------

# Core Idea

The key idea behind an RNN is **memory**.

Every time a new word is processed, the model combines:

-   Current input
-   Previous hidden state (memory)

to produce:

-   New hidden state
-   Output

------------------------------------------------------------------------

# Architecture

              h0
               │
               ▼
    x1 ──► [RNN] ──► h1
                     │
    x2 ──► [RNN] ───► h2
                      │
    x3 ──► [RNN] ───► h3
                       │
    x4 ──► [RNN] ───► h4

Where:

-   x = Input word embedding
-   h = Hidden State (memory)

The same RNN cell is reused for every word in the sequence.

------------------------------------------------------------------------

# Workflow

1.  Convert words into embeddings.
2.  Feed the first embedding to the RNN.
3.  Produce the first hidden state.
4.  Pass the hidden state to the next time step.
5.  Repeat until the final word.
6.  Use the final hidden state (or outputs) for prediction.

Example:

    Sentence
          ↓
    Tokenizer
          ↓
    Embeddings
          ↓
    RNN
          ↓
    Hidden States
          ↓
    Classification / Prediction

------------------------------------------------------------------------

# Applications

-   Sentiment Classification
-   Language Modeling
-   Text Classification
-   Speech Recognition
-   Time-Series Forecasting

------------------------------------------------------------------------

# Advantages

-   Processes sequential data.
-   Remembers previous inputs using the hidden state.
-   Can work with variable-length sequences.
-   More suitable for text than a standard feed-forward network.

------------------------------------------------------------------------

# Limitations

## 1. Vanishing Gradient

During training, gradients become very small over long sequences.

The network gradually forgets information from earlier words.

## 2. Poor Long-Term Memory

RNN performs well on short sequences but struggles with long
dependencies.

Example:

    The movie that I watched last week with my friends was absolutely fantastic.

The model may forget important information from the beginning before
reaching the end.

## 3. Sequential Computation

Words must be processed one after another.

This prevents efficient parallel processing and makes training slower.

------------------------------------------------------------------------

# Historical Significance

    Word2Vec
        ↓
    GloVe
        ↓
    RNN
        ↓
    LSTM

RNN introduced the concept of **sequence modeling** using a hidden
state. It became the foundation for later recurrent architectures.

Its inability to remember information over long sequences led to the
invention of **LSTM**, which introduced dedicated memory cells and
gating mechanisms.

------------------------------------------------------------------------

# Key Takeaways

-   RNN is designed for sequential data.
-   It remembers previous information using a hidden state.
-   It processes one word at a time.
-   It enabled neural sequence modeling.
-   It struggles with long-term dependencies due to the vanishing
    gradient problem.
-   LSTM was introduced to overcome these limitations.
