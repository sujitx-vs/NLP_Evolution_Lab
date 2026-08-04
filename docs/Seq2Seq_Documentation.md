# Seq2Seq (Sequence-to-Sequence)

## What is Seq2Seq?

Seq2Seq is a deep learning architecture that converts one sequence into
another sequence. It is widely used in machine translation,
summarization, chatbots, grammar correction, and question generation.

## Why was it introduced?

RNNs, LSTMs, and GRUs mainly solved many-to-one problems such as
sentiment classification. They could not naturally generate an output
sentence from an input sentence.

Example:

Input:

    I love NLP.

Output:

    J'aime le NLP.

Seq2Seq solved this sequence-to-sequence problem.

## Main Innovation

The major innovation was the **Encoder--Decoder Architecture**.

    Input Sentence
          ↓
    Encoder
          ↓
    Context Vector
          ↓
    Decoder
          ↓
    Output Sentence

Instead of one recurrent network, Seq2Seq uses two: - Encoder:
understands the input sequence. - Decoder: generates the output
sequence.

## Encoder

Reads the input one token at a time and converts the entire sentence
into a fixed-length Context Vector.

## Context Vector

A numerical representation that summarizes the entire input sentence. It
is passed from the encoder to the decoder.

## Decoder

Starts with the Context Vector and generates one output token at a time
until an end token is produced.

## Workflow

1.  Tokenize input.
2.  Convert tokens to embeddings.
3.  Encoder processes the input.
4.  Encoder produces the Context Vector.
5.  Decoder generates the output sequence word by word.

## Applications

-   Machine Translation
-   Text Summarization
-   Chatbots
-   Grammar Correction
-   Question Generation

## Advantages

-   Supports variable-length inputs and outputs.
-   End-to-end neural architecture.
-   Foundation of Neural Machine Translation.

## Shortcomings

The entire input sentence is compressed into one Context Vector.

For long sentences, this creates a **Context Vector Bottleneck**,
causing information loss and poor performance.

## What came next?

The bottleneck led to the **Attention Mechanism**, where the decoder no
longer depends on a single Context Vector but can focus on different
parts of the encoder output while generating each word.

## Historical Position

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
       ↓
    Attention
       ↓
    Transformer
       ↓
    BERT / GPT
