# Gated Recurrent Unit (GRU)

## Introduction

The Gated Recurrent Unit (GRU) is a type of Recurrent Neural Network
(RNN) introduced as a simpler alternative to LSTM. It is designed to
learn long-term dependencies in sequential data while using fewer
parameters and a simpler architecture.

------------------------------------------------------------------------

# Definition

A GRU is a recurrent neural network architecture that uses **Update**
and **Reset** gates to control the flow of information through time.
Unlike LSTM, it does **not** have a separate Cell State.

------------------------------------------------------------------------

# Why was GRU introduced?

LSTM solved the long-term memory problem of RNNs, but it introduced:

-   Three gates
-   A separate Cell State
-   More parameters
-   Higher computational cost

Researchers wanted a model that could achieve similar performance with a
simpler design.

This led to the development of the GRU.

------------------------------------------------------------------------

# Problem GRU Solves

GRU solves the same problem as LSTM:

-   Learning long-term dependencies
-   Reducing the vanishing gradient problem

It does so with a more compact architecture.

------------------------------------------------------------------------

# Core Idea

GRU combines long-term and short-term memory into a single **Hidden
State**.

It uses two gates:

-   **Update Gate** -- Decides how much past information should be kept.
-   **Reset Gate** -- Decides how much previous information should be
    ignored when processing the current input.

------------------------------------------------------------------------

# Architecture

    Previous Hidden State
            │
            ▼
       Update Gate
            │
            ▼
    Input ─► Reset Gate
            │
            ▼
     Candidate Hidden State
            │
            ▼
     New Hidden State

Components:

-   Update Gate
-   Reset Gate
-   Hidden State

There is **no separate Cell State**.

------------------------------------------------------------------------

# Workflow

1.  Convert words into embeddings.
2.  Feed embeddings one by one into the GRU.
3.  Reset Gate decides how much previous information to ignore.
4.  Update Gate decides how much old information to keep.
5.  Compute the new Hidden State.
6.  Use the final Hidden State for prediction.

```{=html}
<!-- -->
```
    Sentence
          ↓
    Tokenizer
          ↓
    Embeddings
          ↓
    GRU
          ↓
    Hidden State
          ↓
    Classification / Prediction

------------------------------------------------------------------------

# Applications

-   Sentiment Analysis
-   Machine Translation
-   Text Classification
-   Speech Recognition
-   Language Modeling
-   Time-Series Forecasting

------------------------------------------------------------------------

# Advantages

-   Simpler than LSTM.
-   Fewer parameters.
-   Faster training.
-   Handles long-term dependencies.
-   Often achieves performance comparable to LSTM.

------------------------------------------------------------------------

# Limitations

## 1. Still Sequential

Processes one time step after another, making parallelization difficult.

## 2. No Separate Cell State

Although simpler, it may be less expressive than LSTM for some complex
tasks.

## 3. Replaced in Many Modern NLP Systems

Large-scale NLP has largely moved to Attention and Transformer-based
architectures.

------------------------------------------------------------------------

# Comparison with LSTM

  Feature          LSTM                    GRU
  ---------------- ----------------------- ---------------
  Gates            Forget, Input, Output   Update, Reset
  Cell State       Yes                     No
  Hidden State     Yes                     Yes
  Parameters       More                    Fewer
  Training Speed   Slower                  Faster
  Complexity       Higher                  Lower

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
        ↓
    Seq2Seq

GRU demonstrated that a simpler recurrent architecture could often
perform as well as LSTM while requiring fewer parameters. It became a
popular choice for many sequence modeling tasks before the widespread
adoption of Attention and Transformers.

------------------------------------------------------------------------

# Key Takeaways

-   GRU is a simplified version of LSTM.
-   Uses two gates: Update and Reset.
-   Has no separate Cell State.
-   Learns long-term dependencies efficiently.
-   Trains faster than LSTM.
-   Was one of the final major recurrent architectures before Seq2Seq
    and Attention became dominant.
