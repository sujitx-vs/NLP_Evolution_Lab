# Long Short-Term Memory (LSTM)

## Introduction

Long Short-Term Memory (LSTM) is a special type of Recurrent Neural
Network (RNN) designed to learn long-term dependencies in sequential
data. It was introduced to overcome the major limitations of standard
RNNs.

------------------------------------------------------------------------

# Definition

An LSTM is a recurrent neural network architecture that uses a **memory
cell** and **gating mechanisms** to decide what information should be
remembered, updated, and forgotten over time.

------------------------------------------------------------------------

# Why was LSTM introduced?

Standard RNNs can process sequences, but they struggle to remember
information over long distances because of the **vanishing gradient
problem**.

Example:

    The movie that I watched last week with my friends was absolutely fantastic.

An RNN may forget important information from the beginning of the
sentence before reaching the end.

LSTM was introduced to preserve important information for much longer
sequences.

------------------------------------------------------------------------

# Problem LSTM Solves

LSTM solves **long-term dependency learning**.

It enables the network to remember useful information from earlier time
steps while discarding irrelevant information.

------------------------------------------------------------------------

# Core Idea

Instead of relying only on a hidden state, LSTM introduces:

-   Cell State (long-term memory)
-   Hidden State (short-term output)

It also uses three gates to control the flow of information.

------------------------------------------------------------------------

# Architecture

                 Previous Cell State
                        │
                        ▼
                 Forget Gate
                        │
                        ▼
    Input ──► Input Gate ──► Cell State ──► Output Gate ──► Hidden State

Components:

-   Forget Gate: Removes unnecessary information.
-   Input Gate: Adds new useful information.
-   Cell State: Stores long-term memory.
-   Output Gate: Produces the output hidden state.

------------------------------------------------------------------------

# Workflow

1.  Convert words into embeddings.
2.  Feed embeddings one by one into the LSTM.
3.  Forget Gate decides what information to remove.
4.  Input Gate decides what new information to store.
5.  Cell State carries long-term information.
6.  Output Gate produces the hidden state.
7.  Final hidden state is used for prediction.

```{=html}
<!-- -->
```
    Sentence
          ↓
    Tokenizer
          ↓
    Embeddings
          ↓
    LSTM
          ↓
    Hidden State
          ↓
    Classification / Prediction

------------------------------------------------------------------------

# Applications

-   Sentiment Analysis
-   Machine Translation
-   Speech Recognition
-   Text Generation
-   Language Modeling
-   Time-Series Forecasting

------------------------------------------------------------------------

# Advantages

-   Learns long-term dependencies.
-   Solves most vanishing gradient issues.
-   Better performance on long sequences.
-   Widely used before Transformers.

------------------------------------------------------------------------

# Limitations

## 1. Complex Architecture

Uses multiple gates and a memory cell, increasing model complexity.

## 2. More Parameters

Requires more memory and computation than a standard RNN.

## 3. Slower Training

Processes sequences one time step at a time, limiting parallelization.

## 4. Still Sequential

Even though it remembers better than an RNN, it still cannot process all
sequence elements simultaneously.

------------------------------------------------------------------------

# Historical Significance

    Word2Vec
        ↓
    GloVe
        ↓
    RNN
        ↓
    LSTM
        ↓
    GRU

LSTM solved the long-term memory problem of RNNs and became the standard
recurrent architecture for many NLP tasks. Later, researchers simplified
LSTM into GRU and eventually moved toward Attention and Transformers.

------------------------------------------------------------------------

# Key Takeaways

-   LSTM is an improved version of RNN.
-   Introduces a Cell State for long-term memory.
-   Uses Forget, Input, and Output gates.
-   Handles long-range dependencies much better than RNN.
-   More accurate but computationally heavier.
-   Inspired the development of GRU and later sequence models.
