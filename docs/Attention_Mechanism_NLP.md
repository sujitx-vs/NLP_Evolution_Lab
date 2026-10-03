# Attention Mechanism — From Seq2Seq Bottleneck to Transformers

## 1. Introduction

Attention is a mechanism that allows a neural network to dynamically determine which parts of available information are most relevant for the current computation.

The core idea is:

> Instead of compressing all information into one fixed representation, allow the model to look back at multiple representations and assign different importance to each of them.

Attention became especially important in Natural Language Processing because language often contains dependencies between words that may be far apart.

For example:

```text
The animal crossed the road because it was tired.
```

To understand the word:

```text
it
```

the model may need information from:

```text
animal
```

Attention provides a direct mechanism for creating such relationships.

---

# 2. Why Attention Was Introduced

To understand Attention, first consider classical Seq2Seq.

A basic LSTM Seq2Seq translation model looks like:

```text
Source Sentence
      ↓
Encoder LSTM
      ↓
Final Hidden State h
Final Cell State c
      ↓
Decoder LSTM
      ↓
Target Sentence
```

For example:

```text
Manglish:
njan nale college il pokum

        ↓

Encoder

        ↓

h_final, c_final

        ↓

Decoder

        ↓

English:
i will go to college tomorrow
```

The problem is that the complete source sentence must be compressed into the final encoder states.

The encoder generates:

```text
h1
h2
h3
h4
h5
```

but basic Seq2Seq mainly passes:

```text
h5, c5
```

to the decoder.

This creates the:

## Fixed Context Bottleneck

A long sentence may contain information about:

- people
- places
- actions
- time
- objects
- relationships
- grammatical structure

Yet all of this must be represented by one fixed-size state.

As source sequences become longer, this becomes difficult.

---

# 3. Main Idea of Attention

Attention keeps all encoder hidden states:

```text
h1, h2, h3, ..., hn
```

and allows the decoder to dynamically decide which ones are useful at each output timestep.

Instead of:

```text
Source
  ↓
one context vector
  ↓
Decoder
```

we now have:

```text
Encoder hidden states

h1
h2
h3
h4
h5
│
│
└──────────► Attention
                ↑
                │
        Decoder current state
```

The decoder asks:

> Which encoder states are most relevant for the word I am generating right now?

---

# 4. Example

Suppose:

```text
Manglish:
njan nale college il pokum
```

Encoder:

```text
njan      → h1
nale      → h2
college   → h3
il        → h4
pokum     → h5
```

Suppose the decoder is currently generating:

```text
tomorrow
```

The model may assign attention weights:

```text
njan       → 0.03
nale       → 0.82
college    → 0.04
il         → 0.02
pokum      → 0.09
```

The decoder mainly focuses on:

```text
nale
```

because it is most relevant to generating:

```text
tomorrow
```

At another timestep, while generating:

```text
college
```

the attention weights may change:

```text
njan       → 0.02
nale       → 0.03
college    → 0.86
il         → 0.05
pokum      → 0.04
```

Attention is therefore:

> dynamic and dependent on the current decoding step.

---

# 5. Basic Attention Pipeline

Attention can be divided into four major operations:

```text
1. Compare
2. Normalize
3. Weight
4. Combine
```

More formally:

```text
Query + Candidates
        ↓
Attention Scores
        ↓
Softmax
        ↓
Attention Weights
        ↓
Weighted Sum
        ↓
Context Vector
```

---

# 6. Attention Scores

Suppose the current decoder state is:

\[
s_t
\]

and encoder hidden states are:

\[
h_1,h_2,\ldots,h_n
\]

Attention calculates:

\[
e_{t,i}
=
score(s_t,h_i)
\]

where:

\[
e_{t,i}
\]

means:

> how relevant is encoder state \(h_i\) to decoder step \(t\)?

Different attention mechanisms mainly differ in how this score is calculated.

---

# 7. Dot-Product Attention

The simplest scoring function is:

\[
e_{t,i}
=
s_t^T h_i
\]

This is simply the dot product between the decoder hidden state and an encoder hidden state.

Example:

\[
s_t=
[1,2]
\]

\[
h_1=[1,1]
\]

\[
h_2=[0,3]
\]

Then:

\[
s_t^T h_1
=
1(1)+2(1)
=
3
\]

and:

\[
s_t^T h_2
=
1(0)+2(3)
=
6
\]

Therefore:

```text
h1 → score 3
h2 → score 6
```

The second encoder state receives the larger compatibility score.

The model learns representations such that relevant states tend to produce useful dot products.

---

# 8. Why Dot Product Can Measure Relevance

For vectors:

```text
similar direction
→ larger positive dot product

weak relationship
→ smaller dot product

opposite direction
→ potentially negative dot product
```

However, Attention does not magically understand semantic relationships from the beginning.

Initially, the representations are mostly untrained.

During training, gradient descent adjusts:

- embeddings
- encoder weights
- decoder weights
- attention-related parameters
- output layers

so that useful representations produce useful attention scores.

---

# 9. Softmax and Attention Weights

Raw attention scores are not probabilities.

Suppose:

```text
scores:

h1 → 1.2
h2 → 4.5
h3 → 0.6
h4 → 2.0
```

Softmax is applied:

\[
\alpha_{t,i}
=
\frac{
e^{e_{t,i}}
}{
\sum_j e^{e_{t,j}}
}
\]

Result:

```text
h1 → 0.03
h2 → 0.85
h3 → 0.02
h4 → 0.10
```

These are the:

## Attention Weights

They satisfy:

\[
0 \leq \alpha_{t,i} \leq 1
\]

and:

\[
\sum_i \alpha_{t,i}=1
\]

---

# 10. Context Vector

The attention weights are used to combine the encoder states:

\[
c_t
=
\sum_i
\alpha_{t,i}h_i
\]

For example:

\[
c_t
=
0.03h_1
+
0.85h_2
+
0.02h_3
+
0.10h_4
\]

The resulting:

\[
c_t
\]

is called the:

## Attention Context Vector

It represents the source information most relevant to the current decoder timestep.

---

# 11. Is the Context Vector the Only Information Passed to the Decoder?

No.

In classical LSTM Seq2Seq with Attention, the encoder's final hidden and cell states are still commonly used to initialize the decoder.

For LSTM:

\[
h_0^{decoder}
=
h_{final}^{encoder}
\]

\[
c_0^{decoder}
=
c_{final}^{encoder}
\]

Then Attention provides an additional context vector at every decoder step:

\[
c_1^{attn},
c_2^{attn},
c_3^{attn},
...
\]

So:

```text
Encoder final h,c
        ↓
initialize Decoder

AND

all encoder hidden states
        ↓
Attention
        ↓
dynamic context vectors
```

Attention supplements the recurrent state rather than necessarily replacing it.

---

# 12. Decoder With Attention

At decoder timestep \(t\):

```text
previous decoder state
        +
current target input
        ↓
decoder hidden state s_t
        ↓
Attention over encoder states
        ↓
context vector c_t
        ↓
combine s_t + c_t
        ↓
output layer
        ↓
next-token probabilities
```

A common approach is concatenation:

\[
[s_t;c_t]
\]

If:

```text
decoder hidden size = 128
context size        = 128
```

then:

```text
combined size = 256
```

This combined representation is passed to the output Dense layer.

---

# 13. Attention Tensor Shapes

Suppose:

```text
Batch size B        = 8
Source length Ts    = 12
Target length Tt    = 10
Hidden size H       = 128
```

Encoder outputs:

\[
(B,T_s,H)
\]

which becomes:

```text
(8,12,128)
```

Decoder outputs:

\[
(B,T_t,H)
\]

which becomes:

```text
(8,10,128)
```

Attention compares every decoder timestep with every encoder timestep.

Attention-score shape:

\[
(B,T_t,T_s)
\]

which becomes:

```text
(8,10,12)
```

Meaning:

```text
8 sentences
×
10 target positions
×
12 source positions
```

---

# 14. Attention Matrix

For one sample, the attention matrix may look like:

| Target / Source | njan | nale | college | il | pokum |
|---|---:|---:|---:|---:|---:|
| i | 0.85 | 0.03 | 0.03 | 0.02 | 0.07 |
| will | 0.10 | 0.10 | 0.05 | 0.05 | 0.70 |
| go | 0.04 | 0.03 | 0.04 | 0.04 | 0.85 |
| college | 0.02 | 0.02 | 0.90 | 0.04 | 0.02 |
| tomorrow | 0.02 | 0.88 | 0.03 | 0.02 | 0.05 |

Rows represent:

```text
decoder / target positions
```

Columns represent:

```text
encoder / source positions
```

This allows Attention to behave somewhat like learned alignment.

---

# 15. Types of Attention

Attention can be categorized in several ways.

These categories are not mutually exclusive.

A system may simultaneously use:

```text
cross-attention
+
global attention
+
soft attention
+
dot-product scoring
```

---

# 16. Dot-Product Attention

Score:

\[
score(q,k)=q^Tk
\]

Advantages:

- simple
- fast
- no additional scoring network
- easily implemented using matrix multiplication

Used heavily in later Transformer architectures.

---

# 17. General / Multiplicative Attention

Instead of directly computing:

\[
q^Tk
\]

use a trainable transformation:

\[
score(q,k)
=
q^TWk
\]

where:

\[
W
\]

is trainable.

This allows the network to learn how the representations should be compared.

Conceptually:

```text
Key
 ↓
learned transformation W
 ↓
compare with Query
 ↓
score
```

---

# 18. Additive Attention

Also strongly associated with Bahdanau-style attention.

Score:

\[
e_{t,i}
=
v^T
\tanh(
W_q q_t
+
W_k k_i
+
b
)
\]

Here:

- \(W_q\) is trainable
- \(W_k\) is trainable
- \(v\) is trainable
- \(b\) is a bias

Conceptually:

```text
Query
  ↓
learned transformation
  ↓
   \
    combine
   /
  ↑
learned transformation
  ↑
Key
```

Then:

```text
tanh
 ↓
learned projection
 ↓
score
```

This uses a small neural scoring network instead of a direct dot product.

---

# 19. Bahdanau Attention

Bahdanau Attention became historically important in neural machine translation.

Its main idea was to let the decoder dynamically align with encoder states rather than relying on one fixed encoder representation.

It is generally associated with:

```text
additive attention
```

---

# 20. Luong Attention

Luong-style Attention explored multiplicative scoring approaches.

Common score functions include:

### Dot

\[
score(q,k)
=
q^Tk
\]

### General

\[
score(q,k)
=
q^TWk
\]

Luong-style Attention is commonly associated with dot-product and multiplicative scoring.

---

# 21. Soft Attention

Soft attention assigns continuous weights to all positions.

Example:

```text
h1 → 0.10
h2 → 0.70
h3 → 0.15
h4 → 0.05
```

Context:

\[
c
=
0.10h_1
+
0.70h_2
+
0.15h_3
+
0.05h_4
\]

Advantages:

- differentiable
- trainable with standard backpropagation
- multiple positions can contribute simultaneously

Most common neural attention mechanisms use soft attention.

---

# 22. Hard Attention

Hard Attention makes a discrete choice.

Example:

```text
h1 → 0
h2 → 1
h3 → 0
h4 → 0
```

Only one location may be selected.

The problem is that discrete choices are difficult to optimize with ordinary gradient descent.

Therefore soft attention became much more common.

---

# 23. Global Attention

Global Attention considers all available positions.

If the source sequence contains:

```text
100 tokens
```

the current query may compare against all 100.

Advantages:

- can access information anywhere

Disadvantages:

- increasingly expensive for long sequences

---

# 24. Local Attention

Local Attention restricts attention to a smaller window.

Instead of:

```text
all 1000 tokens
```

the model might inspect only:

```text
positions 490–510
```

Benefits include lower computational cost.

Local Attention is useful when relevant information is expected to be nearby.

---

# 25. Cross-Attention

Cross-attention occurs when Query and Key/Value representations come from different sequences.

Example:

```text
Decoder
   ↓
Query

Encoder
   ↓
Keys + Values
```

This is what happens in encoder-decoder translation Attention.

For example:

```text
Manglish encoder outputs
        ↓
Keys + Values

English decoder state
        ↓
Query
```

The decoder asks:

> Which parts of the Manglish sentence are relevant to what I am generating right now?

---

# 26. Self-Attention

Self-attention means that a sequence attends to itself.

Suppose:

```text
The animal crossed the road because it was tired
```

For the token:

```text
it
```

the model can compare its representation with representations of:

```text
The
animal
crossed
the
road
because
it
was
tired
```

It may learn a strong relationship between:

```text
it
```

and:

```text
animal
```

Self-attention allows direct relationships between positions within the same sequence.

---

# 27. Did Self-Attention Exist Before Transformers?

Yes.

Attention and self-attention ideas existed before the Transformer architecture.

Transformers did not invent the basic concept of attention.

The major Transformer innovation was:

> making self-attention the central sequence-processing mechanism instead of relying primarily on recurrence.

Before Transformers, attention was frequently added to:

```text
RNNs
LSTMs
GRUs
encoder-decoder models
```

Transformers greatly reduced the role of recurrence.

---

# 28. Encoder Self-Attention

In an encoder using self-attention:

```text
input tokens
↓
each token attends to other input tokens
```

For example:

```text
The bank near the river was flooded.
```

When processing:

```text
bank
```

other words such as:

```text
river
flooded
```

can influence its contextual representation.

---

# 29. Decoder Self-Attention

The decoder can also use self-attention.

Suppose it has generated:

```text
I am going to
```

When calculating the next token, the decoder should use earlier target tokens.

However, during training, the complete target sentence is already available.

Without restrictions, the decoder could look at future words.

That would leak the correct answer.

Therefore decoder self-attention usually uses:

## Causal / Look-Ahead Masking

The current position may attend only to:

```text
itself
and
previous positions
```

not future positions.

---

# 30. Causal Attention

Suppose target tokens are:

```text
I will go home
```

The attention visibility is approximately:

```text
I       → I

will    → I, will

go      → I, will, go

home    → I, will, go, home
```

The model cannot use future tokens.

This preserves autoregressive generation.

---

# 31. Query, Key and Value

Attention is commonly generalized using:

```text
Q = Query
K = Key
V = Value
```

The intuition is similar to information retrieval.

### Query

```text
What information am I looking for?
```

### Key

```text
What does this available item represent?
```

### Value

```text
What information should I retrieve if this item is relevant?
```

---

# 32. Query Is Not Necessarily a Word

Q, K and V are vectors.

They are not directly:

```text
Q = word
K = next word
V = another word
```

Instead:

```text
token
↓
representation
↓
vector
```

The representations are transformed into Query, Key and Value vectors.

---

# 33. Cross-Attention Q, K and V

In encoder-decoder Attention:

```text
Query
≈ decoder representation

Keys
≈ encoder representations

Values
≈ encoder representations
```

The Query is compared against Keys:

\[
QK^T
\]

to determine relevance.

The resulting weights are used to combine Values.

---

# 34. Why Keys and Values Are Different Concepts

Keys determine:

> should I attend to this position?

Values determine:

> what information should I retrieve from this position?

A useful analogy is a dictionary:

```text
Key:
France

Value:
Paris
```

The key is used for matching.

The value contains the information returned.

---

# 35. Self-Attention Q, K and V

Suppose input representations are:

\[
X
\]

Self-attention usually creates:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

where:

\[
W_Q,W_K,W_V
\]

are trainable parameter matrices.

Therefore every token can produce:

```text
Query representation
Key representation
Value representation
```

The same original sequence is used, but each projection has a different purpose.

---

# 36. Self-Attention Example

Sentence:

```text
The cat sat because it was tired
```

Take token:

```text
it
```

Its Query:

```text
Q_it
```

is compared against:

```text
K_The
K_cat
K_sat
K_because
K_it
K_was
K_tired
```

Suppose:

```text
K_cat
```

receives the strongest score.

Attention weights are calculated.

Then the Values:

```text
V_The
V_cat
V_sat
...
```

are combined using those weights.

The resulting contextual representation for:

```text
it
```

now contains information from other relevant words.

---

# 37. Scaled Dot-Product Attention

Transformers use a version of dot-product Attention called:

## Scaled Dot-Product Attention

The equation is:

\[
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}
{\sqrt{d_k}}
\right)V
\]

This equation contains the entire core attention process.

---

# 38. Meaning of \(d_k\)

\[
d_k
\]

is:

> the dimensionality of each Key vector.

Queries normally have the same dimension so that their dot product can be calculated.

Example:

```text
Key dimension = 64
```

Then:

\[
d_k=64
\]

and:

\[
\sqrt{d_k}
=
8
\]

So:

\[
QK^T
\]

is divided by:

\[
8
\]

before softmax.

---

# 39. Why Scale by \(\sqrt{d_k}\)?

As vector dimensionality grows, dot products can become large.

Large scores cause softmax to become extremely sharp.

For example:

```text
scores:
2, 3, 4
```

produce a useful probability distribution.

But:

```text
scores:
20, 30, 40
```

may result in something close to:

```text
0.0000
0.0001
0.9999
```

This can lead to poor gradient behavior.

Scaling:

\[
\frac{1}{\sqrt{d_k}}
\]

keeps score magnitudes more stable.

---

# 40. Full Scaled Dot-Product Process

The complete process is:

```text
Queries
    +
Keys
    ↓

QKᵀ
    ↓

similarity scores
    ↓

divide by √dk
    ↓

scaled scores
    ↓

softmax
    ↓

attention weights
    ↓

multiply by Values
    ↓

contextual representations
```

Mathematically:

\[
A
=
softmax
\left(
\frac{QK^T}
{\sqrt{d_k}}
\right)
\]

Then:

\[
Output
=
AV
\]

---

# 41. Why Multiply by Values?

The attention weights tell us:

```text
how much information to retrieve
from each position
```

Suppose:

```text
position 1 → 0.10
position 2 → 0.75
position 3 → 0.15
```

Then:

\[
Output
=
0.10V_1
+
0.75V_2
+
0.15V_3
\]

Therefore:

```text
Q × K
```

decides **where to look**.

And:

```text
attention weights × V
```

decides **what information to retrieve**.

---

# 42. Attention Is Differentiable

The operations:

```text
matrix multiplication
softmax
weighted sum
```

are differentiable.

Therefore gradients can flow through:

```text
Loss
 ↓
Output layer
 ↓
Attention output
 ↓
Values
Attention weights
 ↓
Queries / Keys
 ↓
Earlier network layers
```

This allows the model to learn where to attend automatically.

---

# 43. Does Attention Have Trainable Parameters?

It depends on the mechanism.

### Simple Dot Product

\[
q^Tk
\]

contains no additional scoring matrix.

However, q and k themselves come from trainable networks.

### General Attention

\[
q^TWk
\]

contains trainable:

\[
W
\]

### Additive Attention

Contains trainable:

\[
W_q,W_k,v,b
\]

### Transformer Attention

Uses trainable projection matrices:

\[
W_Q,W_K,W_V
\]

---

# 44. Main Uses of Attention

Attention has been used in many areas.

## Machine Translation

```text
source language
↓
encoder
↓
attention
↓
decoder
↓
target language
```

This was one of the most important early applications.

---

## Text Summarization

The decoder can focus on important source sentences or tokens while generating a summary.

---

## Question Answering

The model can attend to parts of a passage relevant to the question.

---

## Conversational Systems

Attention helps response generation use relevant parts of previous context.

---

## Speech Recognition

Audio representations can be aligned with generated text tokens.

---

## Image Captioning

Attention can operate over visual regions.

Instead of:

```text
word ↔ word
```

the system may learn:

```text
generated word ↔ image region
```

For example:

```text
"dog"
```

may attend strongly to the image region containing the dog.

---

## Vision

Attention can model relationships between image patches.

This later became central to Vision Transformers.

---

## Multimodal Systems

Attention can connect representations across:

```text
text
images
audio
video
```

Cross-attention is particularly useful for multimodal models.

---

# 45. Advantages of Attention

## Removes the Fixed Context Bottleneck

The decoder no longer depends only on one final encoder state.

It can access all relevant encoder states.

---

## Better Long-Range Dependencies

Attention creates direct relationships between distant positions.

Instead of information travelling through:

```text
h1 → h2 → h3 → h4 → h5
```

Attention can create:

```text
h1 ─────────────► h5
```

---

## Dynamic Context

Each target position gets its own context vector:

\[
c_1,c_2,c_3,\ldots
\]

instead of one fixed representation.

---

## Learned Alignment

Attention can learn relationships between source and target positions.

---

## Improved Interpretability

Attention matrices can be inspected to understand information flow.

However, attention weights should not automatically be treated as complete explanations of model reasoning.

---

# 46. Limitations of Attention

Attention solved important Seq2Seq problems but introduced others.

## Computational Cost

For self-attention with:

\[
N
\]

tokens, every token may compare with every other token.

Approximately:

\[
N^2
\]

pairwise interactions occur.

For:

```text
N = 100
```

this is manageable.

For:

```text
N = 100,000
```

full attention becomes expensive.

---

## Memory Cost

The attention matrix itself can become very large.

Shape:

\[
N\times N
\]

for full self-attention.

---

## Attention Does Not Automatically Understand Order

Attention primarily works through content relationships.

Unlike an RNN:

```text
token1 → token2 → token3
```

self-attention does not inherently know sequence position just from recurrence.

This becomes important when Transformers remove recurrence.

They therefore need another mechanism for representing order.

This leads to:

## Positional Encoding

---

# 47. Attention Did Not Immediately Remove RNNs

Early Attention models were still:

```text
LSTM Encoder
+
Attention
+
LSTM Decoder
```

Attention fixed:

```text
fixed-context bottleneck
```

but RNN/LSTM still caused:

```text
sequential computation
limited parallelization
long recurrent paths
```

The encoder still processed:

```text
token1
 ↓
token2
 ↓
token3
 ↓
token4
```

The decoder still generated:

```text
word1
 ↓
word2
 ↓
word3
```

sequentially.

---

# 48. The Critical Question That Led Toward Transformers

Researchers realized that Attention could directly connect distant positions.

For example:

```text
token1 ───────────────► token20
```

without requiring information to pass through:

```text
token2
token3
...
token19
```

This raised an important question:

> If Attention can directly model relationships between tokens, do we still need recurrence as the main sequence-processing mechanism?

This question led toward the Transformer architecture.

---

# 49. From Seq2Seq to Transformer

The conceptual evolution is:

```text
RNN
↓
sequence processing

LSTM / GRU
↓
better long-term memory

Seq2Seq
↓
encoder + decoder
for variable-length sequence transformation

Problem:
fixed context bottleneck

↓

Seq2Seq + Attention
↓
decoder dynamically accesses encoder states

Remaining problem:
recurrent processing is still sequential

↓

Self-Attention
↓
tokens directly interact with other tokens

↓

Scaled Dot-Product Attention
↓
efficient matrix-based attention

↓

Multi-Head Attention
↓
learn multiple relationships simultaneously

↓

Transformer
↓
attention becomes the central sequence-processing mechanism
```

---

# 50. Attention vs Seq2Seq vs Transformer

These terms represent different concepts.

## LSTM / GRU

Sequence-processing architectures.

## Seq2Seq

Encoder-decoder architecture.

## Attention

Information-selection / information-routing mechanism.

## Self-Attention

Attention where positions inside the same sequence attend to one another.

## Transformer

Architecture built primarily around attention rather than recurrence.

---

# 51. Attention Does Not Mean Transformer

A model can be:

```text
LSTM + Attention
```

without being a Transformer.

For example:

```text
LSTM Encoder
+
LSTM Decoder
+
Bahdanau Attention
```

is an attention-based Seq2Seq model.

It is not a Transformer.

---

# 52. Cross-Attention Does Not Mean Self-Attention

Cross-attention:

```text
Decoder Query
↓
Encoder Keys/Values
```

Self-attention:

```text
Same sequence
↓
Queries
Keys
Values
```

They use similar mathematics but serve different purposes.

---

# 53. Important Attention Terminology

### Query

What information am I looking for?

### Key

Does this available representation match my query?

### Value

What information should I retrieve?

### Score

Compatibility between Query and Key.

### Attention Weight

Normalized importance score after softmax.

### Context Vector

Weighted combination of Values.

### Self-Attention

Q, K and V originate from the same sequence.

### Cross-Attention

Q comes from one sequence/module while K and V come from another.

### Causal Attention

Future positions are masked.

### Global Attention

All available positions can be attended to.

### Local Attention

Only a limited region is considered.

---

# 54. Core Equations

## Dot Product

\[
score(q,k)
=
q^Tk
\]

## General Attention

\[
score(q,k)
=
q^TWk
\]

## Additive Attention

\[
score(q,k)
=
v^T
\tanh(
W_q q
+
W_k k
)
\]

## Softmax

\[
\alpha_i
=
\frac{
e^{score_i}
}{
\sum_j e^{score_j}
}
\]

## Context Vector

\[
c
=
\sum_i
\alpha_i v_i
\]

## Scaled Dot-Product Attention

\[
Attention(Q,K,V)
=
softmax
\left(
\frac{
QK^T
}{
\sqrt{d_k}
}
\right)V
\]

---

# 55. The Most Important Mental Model

Do not memorize Attention only as an equation.

Think:

```text
QUERY
"What am I looking for?"

        ↓

COMPARE AGAINST KEYS

        ↓

SCORES

        ↓

SOFTMAX

        ↓

ATTENTION WEIGHTS

        ↓

USE WEIGHTS TO COMBINE VALUES

        ↓

CONTEXTUAL INFORMATION
```

Or in one sentence:

> Attention compares a Query with available Keys to determine relevance, then uses those relevance weights to combine the corresponding Values.

---

# 56. Classical Seq2Seq Attention Mental Model

```text
              ENCODER

Source
  ↓
h1 h2 h3 h4 h5
│  │  │  │  │
└──┴──┴──┴──┴───────────┐
                         │
                       Keys
                       Values
                         │
                         ▼

                      Attention
                         ▲
                         │
                    Decoder state
                         │
                       Query

                         ↓

                Query × Keys

                         ↓

                       Scores

                         ↓

                       Softmax

                         ↓

                 Attention Weights

                         ↓

              Weighted Sum of Values

                         ↓

                  Context Vector

                         ↓

            Decoder State + Context

                         ↓

                 Output Prediction
```

---

# 57. Transformer Attention Mental Model

Transformer generalizes the same concept:

```text
Input representations X
        │
        ├────► WQ ───► Q
        │
        ├────► WK ───► K
        │
        └────► WV ───► V

Q × Kᵀ
   ↓
divide by √dk
   ↓
softmax
   ↓
attention weights
   ↓
multiply by V
   ↓
new contextual representations
```

Mathematically:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

then:

\[
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

---

# 58. What Attention Solved

Attention addressed one of the main weaknesses of basic Seq2Seq:

```text
Entire input
↓
single fixed vector
```

and replaced it with:

```text
Entire input
↓
multiple representations
↓
dynamic relevance calculation
↓
different context for each computation
```

This improved:

- machine translation
- long-distance relationships
- source-target alignment
- information retrieval within sequences
- contextual representations

---

# 59. What Attention Did Not Solve

Attention alone did not completely solve:

- computational cost for very long sequences
- positional understanding
- autoregressive decoder latency
- memory requirements
- all problems associated with recurrence when used together with RNNs

These limitations motivated further architectural development.

---

# 60. Final Historical Progression

The clean NLP progression is:

```text
Bag of Words / TF-IDF
↓
represent words statistically

Word2Vec / GloVe
↓
learn semantic word representations

RNN
↓
model token order and sequential context

LSTM / GRU
↓
improve long-term dependency handling

Seq2Seq
↓
transform one sequence into another

Problem:
fixed-size encoder context

↓

Attention
↓
dynamically access relevant encoder states

Self-Attention
↓
tokens directly interact with other tokens

Scaled Dot-Product Attention
↓
efficient attention using Q, K and V

Multi-Head Attention
↓
capture multiple relationships simultaneously

Positional Encoding
↓
represent token order without recurrence

Transformer
↓
attention-centric sequence architecture
```

---

# 61. Interview Summary

## What is Attention?

Attention is a neural mechanism that dynamically assigns relevance weights to available representations so that the model can focus on information useful for the current computation.

---

## Why was Attention introduced?

Basic Seq2Seq compressed the whole source sequence into a fixed-size context representation.

Attention allowed the decoder to directly access all encoder states and construct a different context vector at each output timestep.

---

## What is an Attention Score?

It measures compatibility between a Query and a Key.

For dot-product Attention:

\[
score(q,k)=q^Tk
\]

---

## What are Attention Weights?

Softmax-normalized attention scores.

They determine how strongly each Value contributes to the final context.

---

## What is a Context Vector?

A weighted combination of Value vectors:

\[
c=\sum_i\alpha_iV_i
\]

---

## What is Self-Attention?

Self-attention allows positions within the same sequence to attend to one another.

Queries, Keys and Values originate from the same sequence.

---

## What is Cross-Attention?

Cross-attention uses a Query from one sequence or module and Keys/Values from another.

A common example is a decoder attending to encoder outputs.

---

## What is \(d_k\)?

The dimensionality of the Key vectors.

It is used in scaled dot-product Attention:

\[
\frac{QK^T}{\sqrt{d_k}}
\]

to keep dot-product magnitudes stable.

---

## Why did Attention lead to Transformers?

Attention allowed direct relationships between distant tokens.

Researchers realized that these direct interactions could reduce the need for recurrent sequence processing.

Transformers therefore made self-attention the central mechanism and largely removed recurrence.

---

# 62. One-Line Memory Notes

```text
Attention
= dynamically choose relevant information.
```

```text
Query
= what am I looking for?
```

```text
Key
= does this item match what I need?
```

```text
Value
= what information should I retrieve?
```

```text
Attention score
= Query-Key compatibility.
```

```text
Softmax
= converts scores into normalized attention weights.
```

```text
Context
= weighted combination of Values.
```

```text
Self-Attention
= sequence attends to itself.
```

```text
Cross-Attention
= one sequence attends to another.
```

```text
Basic Seq2Seq problem
= fixed context bottleneck.
```

```text
Attention solution
= dynamically access all relevant states.
```

```text
Transformer
= architecture that makes Attention central instead of recurrence.
```

---

# 63. Final Takeaway

The biggest conceptual shift introduced by Attention is:

```text
OLD:

Compress everything
into one representation.

NEW:

Keep multiple representations
and dynamically retrieve
the information needed right now.
```

That principle became one of the foundations of modern deep learning.

The central equation to understand before studying Transformers is:

\[
\boxed{
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
\]

Understanding what happens in each part:

```text
Q
↓
what am I searching for?

K
↓
what information matches?

QKᵀ
↓
compatibility scores

√dk
↓
score stabilization

softmax
↓
attention weights

V
↓
information to retrieve

weights × V
↓
contextual representation
```

provides the direct conceptual foundation for understanding the Transformer architecture.