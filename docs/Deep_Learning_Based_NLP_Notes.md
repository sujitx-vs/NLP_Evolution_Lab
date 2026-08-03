# Deep Learning-based NLP Evolution (Word2Vec → Seq2Seq)

## 1. Word2Vec

### Why?

-   One-hot vectors were sparse and had no semantic meaning.
-   Goal: learn dense word embeddings.

### How?

-   CBOW predicts target from context.
-   Skip-Gram predicts context from target.
-   Train a shallow neural network, update embedding matrix using
    backpropagation.
-   Keep the learned embedding matrix after training.

### Strengths

-   Semantic word vectors.
-   Similar words have similar embeddings.

### Shortcomings

-   Local context only.
-   Static embeddings.
-   Cannot distinguish different meanings of the same word.

## 2. GloVe

### Why?

-   Word2Vec ignores global corpus statistics.

### How?

-   Build a co-occurrence matrix.
-   Learn embeddings from global word co-occurrence counts.

### Strengths

-   Uses global statistics.
-   High-quality static embeddings.

### Shortcomings

-   Still static embeddings.
-   No context awareness.

## 3. RNN

### Why?

-   Needed to understand sequences rather than isolated words.

### How?

-   Processes one token at a time while maintaining a hidden state.

### Strengths

-   Sequence modeling.

### Shortcomings

-   Vanishing gradients.
-   Poor long-term memory.

## 4. LSTM

### Why?

-   Solve RNN's long-term memory problem.

### How?

-   Memory cell with Forget, Input and Output gates.

### Strengths

-   Handles long dependencies.

### Shortcomings

-   More parameters and slower.

## 5. GRU

### Why?

-   Simpler alternative to LSTM.

### How?

-   Uses Update and Reset gates.

### Strengths

-   Faster with fewer parameters.

### Shortcomings

-   Still sequential.

## 6. Seq2Seq

### Why?

-   Needed sequence-to-sequence generation.

### How?

-   Encoder -\> Context Vector -\> Decoder.

### Strengths

-   Translation, summarization, chatbots.

### Shortcomings

-   Context vector bottleneck for long sequences.

## Evolution

1.  Word representation -\> Word2Vec, GloVe
2.  Sequence understanding -\> RNN
3.  Long-term memory -\> LSTM
4.  Simpler recurrent model -\> GRU
5.  Sequence generation -\> Seq2Seq
6.  Next: Attention -\> Transformers.
