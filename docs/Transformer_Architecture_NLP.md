# Transformer Architecture - Complete Notes

## 1. What is a Transformer?

A Transformer is a neural network architecture designed for sequence processing.

It was introduced in the 2017 paper:

```text
Attention Is All You Need
```

The original Transformer was mainly created for:

```text
Machine Translation
```

For example:

```text
English sentence
      |
      v
Transformer
      |
      v
French sentence
```

The major difference between a Transformer and earlier sequence models such as RNN, LSTM, and GRU is:

```text
RNN / LSTM / GRU
    |
Sequential recurrent processing

Transformer
    |
Attention-based processing
```

A Transformer does not need recurrence to process a sequence.

Instead, it mainly uses:

```text
Self-Attention
Multi-Head Attention
Feed Forward Networks
Residual Connections
Layer Normalization
Positional Information
```

---

# 2. Why Was the Transformer Needed?

Before Transformers, sequence-to-sequence tasks commonly used:

```text
RNN
LSTM
GRU
```

A typical machine translation architecture was:

```text
Input sentence
     |
     v
LSTM Encoder
     |
     v
Hidden / Cell State
     |
     v
LSTM Decoder
     |
     v
Translated sentence
```

Later, attention improved this architecture:

```text
Encoder hidden states
        |
        v
Attention
        |
        v
Decoder
```

Attention solved the problem of compressing the whole source sentence into only one final hidden state.

However, one major problem remained.

RNNs and LSTMs still process tokens sequentially.

For example:

```text
I -> love -> machine -> learning
```

The model must calculate:

```text
h1
then h2
then h3
then h4
```

The later state depends on the previous state.

This limits parallelization.

Transformers removed this recurrent dependency.

---

# 3. Main Transformer Idea

Instead of processing:

```text
word 1
then word 2
then word 3
then word 4
```

the Transformer can process the sequence together using self-attention.

Example:

```text
I love machine learning
```

Every token can compare itself with the other tokens.

For example, the token:

```text
machine
```

can directly examine:

```text
I
love
machine
learning
```

and decide how relevant each token is to its representation.

---

# 4. Original Transformer Architecture

The original Transformer is an:

```text
Encoder-Decoder architecture
```

High-level structure:

```text
Source Sentence
      |
      v
Transformer Encoder
      |
      v
Encoder Representations
      |
      v
Transformer Decoder
      |
      v
Target Sentence
```

Example:

```text
Manglish
   |
   v
Encoder
   |
   v
Contextual source representations
   |
   v
Decoder
   |
   v
English
```

---

# 5. Transformer Encoder

The Transformer encoder receives the source sequence.

Example:

```text
njan nale college il pokum
```

First:

```text
Tokens
  |
  v
Token IDs
  |
  v
Token Embeddings
```

But embeddings alone do not tell the Transformer the position of each token.

Therefore positional information is added.

```text
Token Embedding
      +
Positional Representation
      |
      v
Transformer Input Representation
```

Then the representation enters the encoder blocks.

---

# 6. Positional Information

Self-attention does not naturally know sequence order.

For example, without positional information:

```text
dog bites man
```

and:

```text
man bites dog
```

contain the same tokens.

The Transformer therefore needs information about where each token occurs.

For token i:

```text
final_input_i =
token_embedding_i
+
position_embedding_i
```

If:

```text
d_model = 128
```

then:

```text
Token embedding:
(128)

Position representation:
(128)

Combined representation:
(128)
```

The shape does not change.

For a batch:

```text
(B, T, d_model)
```

Example:

```text
(32, 13, 128)
```

---

# 7. How Positional Information Enters Attention

Positional information is not usually passed as a separate input into attention.

Instead:

```text
X = Token Embedding + Position
```

Then Q, K, and V are generated from X.

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Therefore positional information indirectly affects:

```text
Query
Key
Value
```

and consequently affects the attention calculation.

So the flow is:

```text
Token embedding
      +
Position
      |
      v
Combined representation
      |
      v
Q / K / V
      |
      v
Self-Attention
```

---

# 8. Self-Attention

Self-attention means:

```text
A sequence attends to itself.
```

Suppose the sentence is:

```text
I love machine learning
```

Each token creates:

```text
Query
Key
Value
```

For example:

```text
I
 -> Q1
 -> K1
 -> V1

love
 -> Q2
 -> K2
 -> V2

machine
 -> Q3
 -> K3
 -> V3

learning
 -> Q4
 -> K4
 -> V4
```

To calculate the new representation of:

```text
machine
```

its Query is compared with all Keys:

```text
Q_machine dot K_I
Q_machine dot K_love
Q_machine dot K_machine
Q_machine dot K_learning
```

These produce attention scores.

---

# 9. Query, Key, and Value

A useful intuition is:

```text
Query:
What information am I looking for?

Key:
What information does this token represent for matching?

Value:
What information should actually be retrieved?
```

Q, K, and V are produced using learned matrices:

```text
Q = X W_Q

K = X W_K

V = X W_V
```

The matrices:

```text
W_Q
W_K
W_V
```

are trainable parameters.

They are learned through backpropagation.

---

# 10. Scaled Dot-Product Attention

The main attention equation is:

```text
Attention(Q, K, V)
=
softmax(
    (Q K^T) / sqrt(d_k)
)
V
```

This happens in four conceptual steps.

### Step 1 - Compare Query and Key

```text
scores = Q K^T
```

The dot product measures learned compatibility.

---

### Step 2 - Scale the scores

```text
scaled_scores =
scores / sqrt(d_k)
```

Where:

```text
d_k = dimension of each Key vector
```

The scaling helps prevent very large dot-product values.

---

### Step 3 - Apply Softmax

```text
attention_weights =
softmax(scaled_scores)
```

Now the values behave like normalized weights.

Example:

```text
I          0.05
love       0.10
machine    0.25
learning   0.60
```

The weights sum to:

```text
1
```

---

### Step 4 - Weighted combination of Values

```text
output =
attention_weights x V
```

Conceptually:

```text
0.05 * V_I
+
0.10 * V_love
+
0.25 * V_machine
+
0.60 * V_learning
```

The result becomes the contextual representation of the current token.

---

# 11. Attention Matrix

Suppose:

```text
T = 4
```

Then every token compares with every token.

The attention score matrix is:

```text
4 x 4
```

Conceptually:

```text
              I      love    machine   learning

I            ...

love         ...

machine      ...

learning     ...
```

Each row corresponds to one Query token.

Each column corresponds to one Key token.

Therefore:

```text
Attention matrix shape:
(T, T)
```

With batches:

```text
(B, T, T)
```

With multiple heads:

```text
(B, heads, T, T)
```

---

# 12. Multi-Head Attention

Instead of performing only one attention operation, Transformers use multiple attention heads.

Suppose:

```text
d_model = 128
num_heads = 4
```

Then:

```text
d_k = 128 / 4
    = 32
```

The input representation might have shape:

```text
(B, T, 128)
```

After Q, K, V projection and head separation:

```text
(B, 4, T, 32)
```

Each head performs its own attention.

---

# 13. Important Multi-Head Attention Correction

We do NOT do this:

```text
Perform one 128-dimensional attention
        |
        v
split output into 4 heads
```

Instead:

```text
Input representation
        |
        v
Create Q, K, V
        |
        v
Separate into heads
        |
        v
Each head performs attention independently
```

So:

```text
Head 1:
Q1, K1, V1
 -> attention

Head 2:
Q2, K2, V2
 -> attention

Head 3:
Q3, K3, V3
 -> attention

Head 4:
Q4, K4, V4
 -> attention
```

Each head outputs:

```text
(B, T, 32)
```

The heads are then concatenated:

```text
4 x 32 = 128
```

Result:

```text
(B, T, 128)
```

Then another learned projection:

```text
W_O
```

is applied.

---

# 14. Why Multiple Heads?

Different attention heads can learn different relationships.

For example, one head may become useful for:

```text
nearby word relationships
```

another may learn:

```text
long-distance dependencies
```

another may capture:

```text
subject-object relationships
```

another may capture:

```text
semantic relationships
```

These roles are NOT manually assigned.

They emerge through training.

---

# 15. Transformer Encoder Block

One encoder block roughly contains:

```text
Input
  |
  v
Multi-Head Self-Attention
  |
  v
Residual Connection
  |
  v
Layer Normalization
  |
  v
Feed Forward Network
  |
  v
Residual Connection
  |
  v
Layer Normalization
  |
  v
Output
```

---

# 16. Residual Connections

Instead of replacing the input with the attention output:

```text
output = attention(X)
```

the model adds the original representation back:

```text
output =
X + attention(X)
```

This is a residual connection.

It helps:

```text
preserve useful information
improve gradient flow
stabilize deep networks
```

---

# 17. Layer Normalization

After the residual combination, Layer Normalization is used.

Conceptually:

```text
Input
 +
Attention Output
      |
      v
LayerNorm
```

Layer normalization helps stabilize the scale of internal representations during training.

---

# 18. Feed Forward Network

After attention, every token passes through a small feed-forward neural network.

Example:

```text
128
 |
 v
256
 |
activation
 |
 v
128
```

Mathematically:

```text
FFN(x) =
W2 * activation(W1 * x + b1) + b2
```

Important distinction:

```text
Attention:
tokens communicate with other tokens

FFN:
each token representation is transformed independently
```

The same FFN parameters are used at each sequence position.

---

# 19. Encoder Tensor Flow

Suppose:

```text
Batch size = 32
Sequence length = 13
d_model = 128
Heads = 4
d_k = 32
```

Input token IDs:

```text
(32, 13)
```

Embedding + position:

```text
(32, 13, 128)
```

Q, K, V:

```text
(32, 13, 128)
```

Split into heads:

```text
(32, 4, 13, 32)
```

Attention scores:

```text
Q K^T

(32, 4, 13, 13)
```

Attention output per head:

```text
(32, 4, 13, 32)
```

Concatenate:

```text
(32, 13, 128)
```

Feed Forward:

```text
(32, 13, 128)
```

Final encoder representation:

```text
(32, 13, 128)
```

---

# 20. Transformer Decoder

The Transformer decoder contains three important components:

```text
1. Masked Self-Attention
2. Cross-Attention
3. Feed Forward Network
```

Architecture:

```text
Target tokens
     |
     v
Embedding + Position
     |
     v
Masked Multi-Head Self-Attention
     |
     v
Cross-Attention with Encoder
     |
     v
Feed Forward
     |
     v
Final Decoder Representation
```

Residual connections and normalization are also applied around these sublayers.

---

# 21. Why Masked Self-Attention?

During training, the complete target sentence is available.

Example:

```text
sos how are you bro
```

But when predicting:

```text
are
```

the decoder must not see:

```text
you bro
```

because those are future tokens.

Therefore the decoder uses a causal mask.

Conceptually:

```text
             sos   how   are   you   bro

sos           YES   NO    NO    NO    NO

how           YES   YES   NO    NO    NO

are           YES   YES   YES   NO    NO

you           YES   YES   YES   YES   NO

bro           YES   YES   YES   YES   YES
```

This ensures autoregressive prediction.

---

# 22. Decoder Masked Self-Attention

Inside masked self-attention:

```text
Q = decoder representation
K = decoder representation
V = decoder representation
```

So it is still self-attention.

The difference is:

```text
future positions are masked
```

Attention matrix shape:

```text
(B, heads, T_target, T_target)
```

---

# 23. Cross-Attention

After masked self-attention, the decoder needs information from the source sentence.

This is where cross-attention is used.

In cross-attention:

```text
Query:
comes from Decoder

Key:
comes from Encoder

Value:
comes from Encoder
```

This is extremely important.

```text
Decoder Query
      |
      v
compare with
      |
      v
Encoder Keys
      |
      v
retrieve
      |
      v
Encoder Values
```

---

# 24. Cross-Attention Example

Suppose source sequence:

```text
njan nale college il pokum
```

Decoder is currently generating:

```text
tomorrow
```

Its Query may assign attention weights such as:

```text
njan       0.03
nale       0.80
college    0.07
il         0.02
pokum      0.08
```

So the decoder retrieves strong information from:

```text
nale
```

while producing the representation useful for:

```text
tomorrow
```

---

# 25. Cross-Attention Tensor Shapes

Suppose:

```text
Target length = Tt
Source length = Ts
```

Decoder Query:

```text
(B, heads, Tt, d_k)
```

Encoder Key:

```text
(B, heads, Ts, d_k)
```

Then:

```text
Q K^T
```

produces:

```text
(B, heads, Tt, Ts)
```

This means:

```text
each target position
```

can attend to:

```text
every source position
```

---

# 26. Decoder Feed Forward Network

After cross-attention, the representation enters the FFN.

Example:

```text
128
 |
 v
256
 |
ReLU
 |
 v
128
```

The output remains:

```text
(B, T_target, 128)
```

This becomes the final decoder representation.

---

# 27. Vocabulary Projection Layer

The decoder representation itself is not yet a vocabulary prediction.

Suppose:

```text
decoder representation:
128 dimensions
```

and:

```text
target vocabulary:
5000 tokens
```

We need to transform:

```text
128
```

into:

```text
5000 scores
```

A Linear / Dense layer does this.

```text
logits =
decoder_representation * W_vocab + bias
```

Where:

```text
W_vocab shape:
(128, 5000)
```

Therefore:

```text
128-dimensional vector
        |
        v
Linear Vocabulary Projection
        |
        v
5000 logits
```

---

# 28. What is a Linear Layer?

A linear layer is essentially a very shallow neural-network layer.

It performs:

```text
output =
input x weights + bias
```

There is no hidden layer inside it.

For the vocabulary layer:

```text
Input:
decoder representation

Output:
one score for every vocabulary token
```

---

# 29. Logits

The output of the vocabulary projection is called:

```text
logits
```

These are raw scores.

Example:

```text
home       5.8
school     3.4
go         1.1
banana    -0.7
```

Logits are NOT probabilities.

They:

```text
can be negative
can be positive
do not need to sum to 1
```

---

# 30. Softmax and Next Token Probability

Softmax converts logits into probabilities.

Example:

```text
home       0.70
school     0.17
go         0.08
banana     0.01
...
```

The probabilities sum approximately to:

```text
1
```

Now the model has:

```text
P(token | previous tokens, source sentence)
```

For simple greedy decoding:

```text
next_token =
argmax(probabilities)
```

The highest-probability token is selected.

---

# 31. Attention Scores vs Vocabulary Logits

These must not be confused.

Attention scores answer:

```text
Which positions contain relevant information?
```

They come from:

```text
Q K^T / sqrt(d_k)
```

Vocabulary logits answer:

```text
Which token should I generate?
```

They come from:

```text
decoder_representation
x
vocabulary_projection_matrix
```

So the flow is:

```text
Attention scores
      |
      v
Attention weights
      |
      v
Contextual representation
      |
      v
Decoder representation
      |
      v
Vocabulary logits
      |
      v
Softmax
      |
      v
Token probabilities
```

---

# 32. Transformer Training

For translation, training still uses the same shifted-target idea used in Seq2Seq.

Suppose the target is:

```text
how are you bro
```

Training target becomes:

```text
sos how are you bro eos
```

Decoder input:

```text
sos how are you bro
```

Expected output:

```text
how are you bro eos
```

This is equivalent to teacher-forced training.

However, because of masked self-attention, the model can process all target positions in parallel while preventing each position from seeing future tokens.

---

# 33. Training Example

Decoder input:

```text
sos how are you bro
```

Expected targets:

```text
how are you bro eos
```

The model predicts:

```text
Position 1 -> how
Position 2 -> are
Position 3 -> you
Position 4 -> bro
Position 5 -> eos
```

During training, vocabulary logits are calculated for every target position.

Tensor shape:

```text
(B, T_target, V)
```

Where:

```text
B = batch size
T_target = target length
V = target vocabulary size
```

---

# 34. Transformer Inference

During inference, target words are not available.

Generation becomes autoregressive.

Start:

```text
sos
```

Model predicts:

```text
how
```

Next input:

```text
sos how
```

Model predicts:

```text
are
```

Next:

```text
sos how are
```

Then:

```text
you
```

Then:

```text
bro
```

Eventually:

```text
eos
```

---

# 35. LSTM Inference vs Transformer Inference

In LSTM Seq2Seq:

```text
Previous hidden state
Previous cell state
      |
      v
Next decoder step
```

So the decoder carries:

```text
h_t
C_t
```

between timesteps.

In the basic Transformer implementation:

```text
Generated tokens so far
      |
      v
Masked Self-Attention
      |
      v
New contextual representation
```

There is no recurrent hidden state like:

```text
h_t
C_t
```

Instead, the decoder attends over its generated prefix.

---

# 36. Transformer Translation Architecture Used in Our Experiment

Our Manglish -> English model used:

```text
d_model = 128

num_heads = 4

d_k = 32

FFN dimension = 256

Encoder blocks = 2

Decoder blocks = 2
```

Encoder:

```text
Manglish Token IDs
      |
      v
Token Embedding + Position
      |
      v
Encoder Block 1
      |
      v
Encoder Block 2
      |
      v
Encoder Representations
```

Decoder:

```text
English Decoder Input
      |
      v
Token Embedding + Position
      |
      v
Masked Self-Attention
      |
      v
Cross-Attention
      |
      v
FFN
      |
      v
Decoder Representations
      |
      v
Vocabulary Projection
      |
      v
Softmax
      |
      v
English Tokens
```

---

# 37. Comparison With Our LSTM Seq2Seq Experiment

Earlier LSTM Seq2Seq result:

```text
Exact Match Accuracy:
0.6483

Corpus BLEU:
0.7384
```

Transformer result:

```text
Exact Match Accuracy:
0.9125

Corpus BLEU:
0.9509
```

For this learning experiment, the Transformer performed much better on the same translation task.

The important goal of the experiment was not production-level translation.

The goal was to understand the architectural transition:

```text
LSTM recurrence
      |
      v
LSTM + Attention
      |
      v
Transformer Attention Architecture
```

---

# 38. Major Difference From LSTM Seq2Seq

LSTM architecture:

```text
Source tokens
      |
      v
Encoder recurrence
      |
      v
hidden/cell states
      |
      v
Decoder recurrence
```

Transformer:

```text
Source tokens
      |
      v
Self-Attention
      |
      v
Contextual source representations
      |
      v
Cross-Attention
      |
      v
Decoder representation
```

The Transformer does not depend on recurrent state propagation.

---

# 39. Why Transformer Training Can Be More Parallel

An LSTM requires:

```text
h1 -> h2 -> h3 -> h4
```

because:

```text
h3 depends on h2
```

and:

```text
h2 depends on h1
```

Transformer self-attention can calculate relationships between all tokens using matrix operations.

For example:

```text
Q K^T
```

calculates many pairwise relationships simultaneously.

This allows much greater parallelism during training.

---

# 40. Transformer Limitation

Transformer self-attention compares tokens against other tokens.

For sequence length:

```text
T
```

the attention matrix is approximately:

```text
T x T
```

Therefore attention computation and memory can become expensive for very long sequences.

This is often summarized as roughly:

```text
O(T^2)
```

for standard full self-attention.

---

# 41. Transformer Families

The original Transformer uses:

```text
Encoder + Decoder
```

Later models separated these ideas into different architectures.

### Encoder-only

```text
Input
 |
 v
Transformer Encoder
 |
 v
Contextual Representation
```

Commonly used for understanding tasks.

Example family:

```text
BERT
```

---

### Decoder-only

```text
Previous Tokens
      |
      v
Masked Transformer Decoder
      |
      v
Next Token
```

Commonly used for language modeling and generation.

Example family:

```text
GPT
```

---

### Encoder-Decoder

```text
Input Sequence
      |
      v
Encoder
      |
      v
Decoder
      |
      v
Output Sequence
```

Commonly used for sequence transformation.

Examples include:

```text
translation
summarization
text-to-text tasks
```

---

# 42. Historical Progression

A useful NLP progression is:

```text
RNN
 |
 v
LSTM / GRU
 |
 v
Seq2Seq
 |
 v
Seq2Seq + Attention
 |
 v
Self-Attention
 |
 v
Transformer
 |
 +--------------------------+
 |                          |
 v                          v
Encoder models         Decoder models

BERT-like              GPT-like

          +
          |
          v

Encoder-Decoder models

T5-like
```

---

# 43. Core Transformer Equations

## Positional input

```text
X =
TokenEmbedding + Position
```

## Query

```text
Q =
X W_Q
```

## Key

```text
K =
X W_K
```

## Value

```text
V =
X W_V
```

## Attention scores

```text
Scores =
Q K^T
```

## Scaled scores

```text
ScaledScores =
Scores / sqrt(d_k)
```

## Attention weights

```text
Weights =
softmax(ScaledScores)
```

## Attention output

```text
AttentionOutput =
Weights V
```

## Multi-head attention

```text
MultiHead =
Concat(
    Head1,
    Head2,
    ...,
    HeadN
)
W_O
```

## Feed Forward Network

```text
FFN(x) =
W2 * activation(W1 * x + b1) + b2
```

## Vocabulary logits

```text
Logits =
DecoderRepresentation
*
W_vocab
+
b_vocab
```

## Token probabilities

```text
Probabilities =
softmax(Logits)
```

---

# 44. Complete Transformer Flow

```text
SOURCE TOKENS
      |
      v
Source Token Embeddings
      +
Source Position Information
      |
      v
Multi-Head Self-Attention
      |
      v
Residual + LayerNorm
      |
      v
Feed Forward Network
      |
      v
Residual + LayerNorm
      |
      v
ENCODER REPRESENTATIONS
      |
      |
      |--------------------------------+
                                       |
                                       v
TARGET TOKENS                      Cross-Attention
      |                                ^
      v                                |
Target Token Embeddings                |
      +                                |
Target Position Information            |
      |                                |
      v                                |
Masked Multi-Head Self-Attention ------+
      |
      v
Residual + LayerNorm
      |
      v
Cross-Attention
      |
      v
Residual + LayerNorm
      |
      v
Feed Forward Network
      |
      v
Residual + LayerNorm
      |
      v
FINAL DECODER REPRESENTATION
      |
      v
Linear Vocabulary Projection
      |
      v
LOGITS
      |
      v
Softmax
      |
      v
TOKEN PROBABILITIES
      |
      v
NEXT TOKEN
```

---

# 45. Final Mental Model

The Transformer encoder answers:

```text
How should every source token be represented
after considering the complete source sequence?
```

Self-attention answers:

```text
Which tokens in my own sequence are relevant
to this token?
```

Masked self-attention answers:

```text
Which previous target tokens are relevant,
without looking into the future?
```

Cross-attention answers:

```text
Which parts of the source sequence are relevant
while generating this target representation?
```

The FFN answers:

```text
How should this contextual token representation
be transformed internally?
```

The vocabulary projection answers:

```text
Given the final decoder representation,
what raw score should every vocabulary token receive?
```

Softmax answers:

```text
What probability distribution should those
vocabulary scores become?
```

And autoregressive generation answers:

```text
Which token should be generated next?
```

---

# 46. One-Line Summary

```text
Transformer =
Positional Information
+
Self-Attention
+
Multi-Head Attention
+
Cross-Attention
+
Feed Forward Networks
+
Residual Connections
+
Layer Normalization
+
Vocabulary Prediction
```

The major architectural idea is:

```text
Instead of carrying information sequentially
through recurrent hidden states,

allow tokens to directly attend to other tokens
and build contextual representations through attention.
```