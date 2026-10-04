# Attention Mechanism in NLP

## 1. Introduction

Attention is a mechanism that allows a neural network to dynamically decide which parts of the available information are most relevant for the current computation.

It became especially important in sequence-to-sequence tasks such as machine translation.

Before attention, classical Seq2Seq models often used:

```text
Input Sequence
    ->
Encoder RNN / LSTM / GRU
    ->
Final Hidden State
    ->
Decoder RNN / LSTM / GRU
    ->
Output Sequence
```

The major problem was that the entire input sequence had to be compressed into a single fixed-size representation.

Attention was introduced to remove this bottleneck.

With attention:

```text
Input Sequence
    ->
Encoder
    ->
All Encoder Hidden States
    ->
Attention
    ->
Decoder
    ->
Output Sequence
```

The decoder can dynamically access different encoder states while generating each output token.

---

# 2. Why Attention Was Needed

## 2.1 Basic Seq2Seq Architecture

Consider a source sentence:

```text
njan nale kozhikode pokum
```

Suppose the encoder produces:

```text
h1
h2
h3
h4
h5
```

In a basic Seq2Seq model, the decoder mainly depends on the final encoder states:

```text
h5
c5
```

For an LSTM, these correspond to:

```text
Final Hidden State
Final Cell State
```

The entire source sentence must therefore be represented inside these final states.

This works reasonably well for short sequences.

For longer sequences, it becomes difficult.

---

## 2.2 Fixed Context Bottleneck

Suppose the sentence is:

```text
the student who came from france yesterday gave the professor a book
```

The encoder has to remember information about:

```text
student
france
yesterday
professor
book
```

and the relationships between them.

In basic Seq2Seq:

```text
Entire Input
    ->
One Final Representation
    ->
Decoder
```

This is called the fixed-context bottleneck.

Attention changes this to:

```text
Encoder Hidden States

h1
h2
h3
h4
h5
...

Decoder can access all of them
```

---

# 3. Core Idea of Attention

At each decoder timestep, the decoder asks:

```text
Which encoder hidden states are most relevant for generating the current output token?
```

Suppose the encoder produced:

```text
h1 h2 h3 h4 h5
```

and the decoder currently has hidden state:

```text
s_t
```

Attention calculates a relevance score between:

```text
s_t
```

and every encoder hidden state:

```text
h1
h2
h3
h4
h5
```

This gives:

```text
e1
e2
e3
e4
e5
```

These scores are then converted into attention weights.

---

# 4. Basic Attention Pipeline

The attention mechanism can be understood in four steps.

```text
1. Compare
2. Normalize
3. Weight
4. Combine
```

More specifically:

```text
Decoder State
    +
Encoder States
    ->
Attention Scores
    ->
Softmax
    ->
Attention Weights
    ->
Weighted Sum of Encoder States
    ->
Context Vector
```

---

# 5. Attention Score

The attention score measures the compatibility between the current decoder state and an encoder state.

General form:

```text
score(decoder_state, encoder_state)
```

For decoder timestep `t` and encoder position `i`:

```text
e_ti = score(s_t, h_i)
```

where:

```text
s_t = current decoder state

h_i = encoder hidden state at source position i

e_ti = raw attention score
```

The score is not yet a probability.

---

# 6. Dot Product Attention

One simple scoring method is dot product attention.

Formula:

```text
e_ti = s_t^T h_i
```

This means taking the dot product between:

```text
Decoder Hidden State
and
Encoder Hidden State
```

Example:

```text
s_t = [1, 2]

h1 = [1, 1]
h2 = [0, 3]
h3 = [-1, 0]
```

Scores:

```text
score(s_t, h1)

= 1*1 + 2*1
= 3
```

```text
score(s_t, h2)

= 1*0 + 2*3
= 6
```

```text
score(s_t, h3)

= 1*(-1) + 2*0
= -1
```

Therefore:

```text
h1 -> 3
h2 -> 6
h3 -> -1
```

The second encoder state gets the highest score.

---

# 7. Why Dot Product Works

The dot product provides a measure of compatibility between two vectors.

Rough intuition:

```text
Similar direction
-> larger positive score

Weak relationship
-> score closer to zero

Opposite direction
-> negative score
```

The network learns useful representations during training.

The vectors are not meaningful at initialization.

Training changes:

```text
Embeddings
Encoder Weights
Decoder Weights
Attention Representations
Output Weights
```

so that relevant encoder and decoder states produce useful attention scores.

---

# 8. Softmax and Attention Weights

Raw attention scores might look like:

```text
3
6
-1
```

Softmax converts them into normalized values.

Example:

```text
3
6
-1

-> softmax

0.047
0.952
0.001
```

The resulting values are called attention weights.

They are commonly written as:

```text
alpha_ti
```

The weights satisfy:

```text
0 <= alpha_ti <= 1
```

and:

```text
sum(alpha_ti) = 1
```

These values describe how strongly the model is attending to each position.

---

# 9. Context Vector

After calculating the attention weights, the model creates a context vector.

Formula:

```text
c_t = sum(alpha_ti * h_i)
```

Example:

```text
alpha1 = 0.05
alpha2 = 0.80
alpha3 = 0.15
```

Then:

```text
c_t =
0.05 * h1
+
0.80 * h2
+
0.15 * h3
```

The result is a weighted combination of the encoder states.

This is called the attention context vector.

---

# 10. Why Use a Weighted Sum

Instead of selecting only one encoder state, attention normally combines multiple states.

Example:

```text
puthiya red car
```

A generated output word might depend on information from:

```text
puthiya
red
car
```

at the same time.

Soft attention can combine all of these.

This also keeps the process differentiable, allowing training with backpropagation.

---

# 11. Attention Context Changes at Every Decoder Step

This is one of the main differences between basic Seq2Seq and attention-based Seq2Seq.

Basic Seq2Seq:

```text
One Source Representation
    ->
Used for the Entire Output Sequence
```

Attention-based Seq2Seq:

```text
Decoder Step 1 -> Context c1

Decoder Step 2 -> Context c2

Decoder Step 3 -> Context c3

Decoder Step 4 -> Context c4
```

Each decoder timestep gets its own source context.

---

# 12. Example of Dynamic Attention

Source:

```text
njan nale kozhikode pokum
```

Suppose the English translation is:

```text
i will go to kozhikode tomorrow
```

When generating:

```text
i
```

attention may focus strongly on:

```text
njan
```

When generating:

```text
kozhikode
```

attention may focus strongly on:

```text
kozhikode
```

When generating:

```text
tomorrow
```

attention may focus strongly on:

```text
nale
```

Therefore, the attention distribution changes for every generated token.

---

# 13. Encoder States and Decoder States With Attention

Attention does not necessarily replace the normal Seq2Seq state transfer.

For an LSTM encoder:

```text
Final Encoder Hidden State
Final Encoder Cell State
```

can still initialize the decoder.

Example:

```text
decoder_h0 = encoder_final_h

decoder_c0 = encoder_final_c
```

Attention then provides additional information at every decoder timestep.

Conceptually:

```text
Encoder Final h and c
    ->
Initialize Decoder

All Encoder Hidden States
    ->
Attention
    ->
Context Vector for Every Decoder Step
```

---

# 14. Encoder Output With Attention

Without attention, the encoder may only need the final state.

With attention, we need all encoder hidden states.

For example:

```text
Input Length = 10
Hidden Size = 128
Batch Size = 32
```

Encoder outputs:

```text
(32, 10, 128)
```

Meaning:

```text
32 samples
10 source positions
128 hidden values per position
```

In Keras, this normally requires:

```python
LSTM(
    128,
    return_sequences=True,
    return_state=True
)
```

The model now keeps:

```text
All Hidden States
Final Hidden State
Final Cell State
```

---

# 15. Attention Matrix

Suppose:

```text
Source Length = 5
Target Length = 4
```

The attention matrix for one sample may have shape:

```text
(4, 5)
```

Example:

| Target / Source | njan | nale | college | il | pokum |
|---|---:|---:|---:|---:|---:|
| i | 0.85 | 0.03 | 0.03 | 0.02 | 0.07 |
| will | 0.10 | 0.10 | 0.05 | 0.05 | 0.70 |
| go | 0.05 | 0.05 | 0.05 | 0.05 | 0.80 |
| tomorrow | 0.03 | 0.85 | 0.03 | 0.02 | 0.07 |

Each row corresponds to one decoder position.

Each column corresponds to one source position.

---

# 16. Tensor Shapes in Attention

Suppose:

```text
Batch Size = B
Source Length = Ts
Target Length = Tt
Hidden Size = H
```

Encoder states:

```text
(B, Ts, H)
```

Decoder states:

```text
(B, Tt, H)
```

Attention scores:

```text
(B, Tt, Ts)
```

For example:

```text
B  = 8
Ts = 12
Tt = 10
H  = 128
```

Then:

```text
Encoder Outputs:
(8, 12, 128)

Decoder Outputs:
(8, 10, 128)

Attention Scores:
(8, 10, 12)
```

Meaning:

```text
8 samples
10 target positions
12 source positions to attend to
```

---

# 17. Main Types of Attention

There are several attention variants.

They can be classified by:

```text
How the score is calculated

Where Query, Key, and Value come from

How many positions can be attended to

Whether selection is soft or hard
```

---

# 18. Dot Product Attention

Formula:

```text
score(q, k) = q^T k
```

Advantages:

```text
Simple
Fast
Easy to compute using matrix multiplication
```

The decoder state can act as the query.

Encoder states act as keys.

---

# 19. General or Multiplicative Attention

Formula:

```text
score(q, k) = q^T W k
```

where:

```text
W
```

is a trainable matrix.

Instead of directly comparing the vectors, the model learns a transformation.

Conceptually:

```text
Key
 ->
Learned Transformation
 ->
Compare With Query
 ->
Score
```

---

# 20. Additive Attention

Additive attention is strongly associated with Bahdanau attention.

A common form is:

```text
score(q, k)
=
v^T tanh(Wq q + Wk k + b)
```

Here:

```text
Wq
Wk
v
b
```

are trainable parameters.

Conceptually:

```text
Query
 ->
Learned Transformation
       \
        -> Combine -> tanh -> Score
       /
Key
 ->
Learned Transformation
```

This allows the network to learn how query and key representations should be compared.

---

# 21. Bahdanau Attention

Bahdanau attention was one of the important early neural attention mechanisms used in machine translation.

It is usually associated with additive attention.

It allowed the decoder to dynamically examine encoder states instead of depending on one fixed source vector.

Main benefit:

```text
Better handling of longer source sequences
```

---

# 22. Luong Attention

Luong-style attention explored several scoring methods.

Examples include:

```text
Dot:

score(q, k) = q^T k
```

```text
General:

score(q, k) = q^T W k
```

Luong-style attention is often associated with multiplicative attention.

---

# 23. Soft Attention

Soft attention assigns a continuous weight to every available position.

Example:

```text
0.05
0.10
0.75
0.10
```

Then the context is calculated using a weighted sum.

Advantages:

```text
Differentiable
Works directly with backpropagation
Can use information from multiple positions
```

Most common attention mechanisms used in NLP are soft attention.

---

# 24. Hard Attention

Hard attention chooses one or a small number of positions.

Example:

```text
0
0
1
0
```

This means:

```text
Select only the third position
```

Hard selection is more difficult to train because discrete choices are not naturally differentiable.

---

# 25. Global Attention

Global attention allows the current query to attend to all available positions.

Example:

```text
Source Length = 100

Query compares with all 100 positions
```

Advantages:

```text
Can retrieve information from anywhere in the sequence
```

Disadvantage:

```text
More computationally expensive for long sequences
```

---

# 26. Local Attention

Local attention only considers a limited region.

Example:

```text
Current Position = 50

Attend only to positions:

45 to 55
```

Advantages:

```text
Reduced computation
Useful when important information is expected nearby
```

---

# 27. Cross-Attention

Cross-attention means that Query comes from one sequence while Keys and Values come from another sequence.

In Seq2Seq:

```text
Decoder
    ->
Query

Encoder
    ->
Keys and Values
```

Conceptually:

```text
Decoder asks:

"What source information do I need now?"
```

Then it searches through the encoder representations.

Your Manglish-to-English LSTM attention model is an example of cross-attention.

---

# 28. Self-Attention

Self-attention means a sequence attends to itself.

Example:

```text
the animal crossed the road because it was tired
```

When processing:

```text
it
```

the model may attend strongly to:

```text
animal
```

All Query, Key, and Value representations are derived from the same sequence.

Conceptually:

```text
Token 1
Token 2
Token 3
Token 4
...

Each token can examine other tokens
```

---

# 29. Self-Attention Existed Before Transformers

Self-attention was explored before Transformers.

The major Transformer innovation was not simply inventing attention.

The important architectural change was:

```text
Make attention the central sequence-processing mechanism
instead of relying mainly on recurrence.
```

Before Transformers, systems could combine:

```text
RNN / LSTM
+
Attention
```

Transformers moved toward:

```text
Attention
+
Feed Forward Layers
+
Residual Connections
+
Normalization
```

without recurrent sequence processing as the central mechanism.

---

# 30. Encoder Self-Attention

In a Transformer-style encoder:

```text
Input Tokens
    ->
Self-Attention
```

Each source token can attend to other source tokens.

Example:

```text
the animal did not cross the road because it was tired
```

The representation of:

```text
it
```

can incorporate information from:

```text
animal
```

This produces contextual representations.

---

# 31. Decoder Self-Attention

The decoder also uses self-attention.

However, during autoregressive generation, it must not look at future output tokens.

Suppose the output currently is:

```text
i am going
```

The decoder can attend to:

```text
i
am
going
```

but not future tokens that have not been generated yet.

This requires masking.

---

# 32. Masked Self-Attention

Consider:

```text
Token 1
Token 2
Token 3
Token 4
```

When computing Token 2:

```text
Allowed:

Token 1
Token 2
```

Not allowed:

```text
Token 3
Token 4
```

A causal mask blocks future positions.

This prevents information leakage during training.

---

# 33. Query, Key, and Value

Attention is often generalized using:

```text
Q = Query
K = Key
V = Value
```

The intuition is similar to searching a database.

```text
Query:
What am I looking for?

Key:
Does this item match what I am looking for?

Value:
What information should I retrieve from the item?
```

---

# 34. Query

The Query represents what the current position is searching for.

In classical encoder-decoder attention:

```text
Decoder State
->
Query
```

Example:

The decoder is generating the word corresponding to:

```text
tomorrow
```

Its current state produces a Query that may strongly match the encoder representation corresponding to:

```text
nale
```

---

# 35. Key

The Key is used for matching.

Each available position provides a Key.

The Query is compared with the Keys.

Example:

```text
Query
    |
    +-> Key 1
    +-> Key 2
    +-> Key 3
    +-> Key 4
```

This produces attention scores.

The Key does not mean:

```text
next word
```

It is simply the representation used to decide whether a position is relevant.

---

# 36. Value

The Value contains the information that will actually be retrieved.

The Query and Key determine:

```text
How much attention should this position receive?
```

The Value determines:

```text
What information should be taken from that position?
```

The attention weights are applied to the Values.

---

# 37. Why Separate Key and Value

Conceptually:

```text
Key
=
used for matching

Value
=
used for retrieving information
```

Example:

```text
Database Key:
France

Database Value:
Paris
```

You search using the Key.

You retrieve the Value.

Attention uses a similar concept.

---

# 38. Q, K, and V in Classical Cross-Attention

In a simple Seq2Seq attention model:

```text
Q
=
Decoder Hidden State

K
=
Encoder Hidden States

V
=
Encoder Hidden States
```

Keys and Values may come from the same encoder representations.

Their conceptual roles are still different.

---

# 39. Q, K, and V in Self-Attention

Suppose the sequence representation is:

```text
X
```

Self-attention creates:

```text
Q = X W_Q

K = X W_K

V = X W_V
```

where:

```text
W_Q
W_K
W_V
```

are learned projection matrices.

Therefore, every token representation is transformed into:

```text
Query representation

Key representation

Value representation
```

---

# 40. Self-Attention Example

Sentence:

```text
the cat slept because it was tired
```

When computing the representation for:

```text
it
```

the Query for `it` is compared against Keys for:

```text
the
cat
slept
because
it
was
tired
```

Suppose:

```text
cat
```

gets the highest attention weight.

The new representation for `it` therefore receives significant information from the Value associated with `cat`.

---

# 41. Scaled Dot Product Attention

Transformers use scaled dot product attention.

Formula:

```text
Attention(Q, K, V)
=
softmax(
    (Q K^T) / sqrt(d_k)
) V
```

This contains several stages.

First:

```text
Q K^T
```

calculates compatibility scores.

Then:

```text
divide by sqrt(d_k)
```

scales the scores.

Then:

```text
softmax
```

creates attention weights.

Finally:

```text
multiply by V
```

retrieves and combines the information.

---

# 42. What is d_k?

`d_k` is the dimensionality of each Key vector.

Example:

```text
Key Dimension = 64
```

Then:

```text
d_k = 64
```

and:

```text
sqrt(d_k) = 8
```

Therefore:

```text
Q K^T
```

is divided by:

```text
8
```

before softmax.

---

# 43. Why Scale the Dot Product

With high-dimensional vectors, dot products can become large.

Large scores can make softmax extremely sharp.

Example:

```text
20
30
40
```

may produce something close to:

```text
0.0000
0.0001
0.9999
```

This can lead to poor gradient behavior.

Scaling by:

```text
sqrt(d_k)
```

keeps the values more stable.

---

# 44. Attention Matrix in Self-Attention

Suppose a sequence contains:

```text
4 tokens
```

Then every token may compare with every token.

The attention matrix can be:

```text
4 x 4
```

Example:

| Query / Key | the | cat | slept | tired |
|---|---:|---:|---:|---:|
| the | 0.4 | 0.2 | 0.2 | 0.2 |
| cat | 0.1 | 0.6 | 0.2 | 0.1 |
| slept | 0.1 | 0.4 | 0.4 | 0.1 |
| tired | 0.1 | 0.5 | 0.1 | 0.3 |

Each row describes where one token attends.

---

# 45. Attention Complexity

For full self-attention with sequence length:

```text
N
```

every token can interact with every other token.

Therefore, the number of pairwise comparisons grows roughly as:

```text
N^2
```

Example:

```text
N = 10

Approximate interactions = 100
```

```text
N = 1000

Approximate interactions = 1,000,000
```

This makes full attention expensive for very long sequences.

---

# 46. Main Advantages of Attention

## Better Handling of Long Sequences

The decoder can directly access earlier source representations.

It does not depend entirely on information surviving inside one final encoder state.

---

## Dynamic Context

Different output positions can use different source information.

---

## Better Alignment

Attention naturally learns relationships such as:

```text
njan -> i

nale -> tomorrow

pokum -> go
```

This was especially valuable in machine translation.

---

## Better Gradient Paths

A relevant source representation can directly influence a later output through attention.

This can reduce the effective distance information must travel compared with purely recurrent systems.

---

## Interpretability

Attention matrices can be visualized.

This can help inspect which positions are being weighted strongly.

However, attention weights should not automatically be treated as complete explanations of model reasoning.

---

# 47. Main Limitations of Attention

Attention does not solve everything.

## Computational Cost

Full self-attention has roughly quadratic sequence complexity.

---

## Attention Weights Are Not Perfect Explanations

A high attention score means that information receives more weight inside the attention computation.

It does not automatically mean:

```text
"This is the exact reason the model made the decision."
```

---

## Early Attention Models Still Used Recurrence

Models such as:

```text
LSTM Encoder
+
Attention
+
LSTM Decoder
```

still processed sequences recurrently.

Therefore:

```text
Token 1
->
Token 2
->
Token 3
```

remained sequential.

This limited parallel processing.

---

# 48. Attention Use Cases

Attention has been used in many domains.

## Machine Translation

```text
English
->
French
```

Attention helps align target words with relevant source words.

---

## Text Summarization

The decoder can attend to important parts of the source document while generating a summary.

---

## Question Answering

The model can focus on relevant parts of a passage while answering a question.

---

## Dialogue Systems

The response generator can attend to relevant parts of the conversation history.

---

## Speech Recognition

Attention can align audio representations with output text.

---

## Image Captioning

A caption generator can attend to different image regions while generating different words.

Example:

```text
dog
```

may attend to the dog region.

```text
ball
```

may attend to the ball region.

---

## Vision Models

Attention can model relationships between image patches.

Modern Vision Transformers use this idea.

---

## Multimodal Models

Attention can connect information across:

```text
Text
Images
Audio
Video
```

Cross-attention is especially useful in multimodal systems.

---

# 49. Attention Before Transformers

The historical progression can be simplified as:

```text
RNN
    ->
LSTM / GRU
    ->
Encoder-Decoder Seq2Seq
    ->
Seq2Seq + Attention
    ->
Self-Attention
    ->
Transformer
```

The important conceptual progression was:

```text
First:
Remember the sequence using recurrence.

Then:
Separate input encoding and output generation.

Then:
Allow the decoder to directly inspect encoder states.

Then:
Allow tokens to attend directly to other tokens.

Finally:
Make attention the main sequence-processing mechanism.
```

---

# 50. Why Attention Led to Transformers

Attention created direct connections between distant positions.

In an RNN:

```text
Token 1
 ->
h1
 ->
h2
 ->
h3
 ->
h4
 ->
Token 5
```

Information may need to travel through multiple recurrent steps.

Attention allows:

```text
Token 1
----------------->
Token 5
```

directly.

Researchers therefore asked:

```text
If attention can directly model relationships between tokens,
do we still need recurrence as the central sequence mechanism?
```

This idea led toward Transformer architectures.

---

# 51. Attention in the Transformer Encoder

The Transformer encoder uses self-attention.

Conceptually:

```text
Input Tokens
    ->
Embeddings
    ->
Positional Information
    ->
Self-Attention
    ->
Feed Forward Network
    ->
Contextual Representations
```

Each source token can access information from other source tokens.

---

# 52. Attention in the Transformer Decoder

A Transformer decoder typically contains two important attention mechanisms.

## Masked Self-Attention

The target sequence attends to previously generated target tokens.

```text
Current Target Position
    ->
Previous Target Positions
```

Future positions are masked.

---

## Cross-Attention

The decoder then attends to encoder representations.

```text
Decoder
    ->
Queries

Encoder
    ->
Keys and Values
```

This allows the decoder to retrieve relevant source information.

---

# 53. Why Transformers Need Positional Information

RNNs naturally process tokens in order.

Example:

```text
Token 1
->
Token 2
->
Token 3
```

Self-attention does not inherently know sequence order.

Therefore Transformers add positional information so the model can distinguish:

```text
first token
second token
third token
...
```

This is handled using positional encodings or learned positional representations.

---

# 54. Multi-Head Attention

Instead of performing only one attention operation, Transformers use multiple attention heads.

Conceptually:

```text
Head 1 -> one relationship pattern
Head 2 -> another relationship pattern
Head 3 -> another relationship pattern
...
```

Each head has its own learned Q, K, and V projections.

The results are combined afterward.

This allows the model to represent multiple relationships simultaneously.

For example, different heads may specialize in patterns involving:

```text
Syntax
Pronouns
Long-distance relationships
Nearby dependencies
Semantic associations
```

The exact behavior is learned, not manually assigned.

---

# 55. Single-Head vs Multi-Head Attention

Single-head attention:

```text
Q
K
V
    ->
One Attention Calculation
```

Multi-head attention:

```text
Q K V
 -> Head 1

Q K V
 -> Head 2

Q K V
 -> Head 3

...

Combine Heads
```

This increases representation capacity.

---

# 56. Main Attention Categories Summary

| Type | Main Idea |
|---|---|
| Dot Product | Compare vectors using q^T k |
| General / Multiplicative | Use q^T W k |
| Additive / Bahdanau | Use a learned neural scoring function |
| Soft Attention | Continuous weights over positions |
| Hard Attention | Discrete position selection |
| Global Attention | Attend to all positions |
| Local Attention | Attend to a limited region |
| Cross-Attention | Query and Key/Value come from different sequences |
| Self-Attention | Query, Key, and Value come from the same sequence |
| Masked Self-Attention | Self-attention with future positions hidden |
| Scaled Dot Product | Dot product divided by sqrt(d_k) |
| Multi-Head Attention | Multiple attention operations in parallel |

---

# 57. Architecture Classification of the Manglish Model

The Manglish-to-English model developed during this learning experiment can be described as:

```text
LSTM Encoder
+
LSTM Decoder
+
Global Attention
+
Soft Attention
+
Cross-Attention
+
Dot Product Scoring
```

The encoder produces:

```text
All Hidden States
+
Final Hidden State
+
Final Cell State
```

The decoder is initialized using:

```text
Final Encoder h
Final Encoder c
```

Attention then dynamically retrieves information from all encoder hidden states during output generation.

---

# 58. Important Distinctions

## LSTM

```text
A recurrent sequence-processing architecture.
```

---

## Seq2Seq

```text
An encoder-decoder architecture for transforming one sequence into another.
```

---

## Attention

```text
A mechanism that dynamically retrieves relevant information from available representations.
```

---

## Self-Attention

```text
Attention where a sequence attends to itself.
```

---

## Transformer

```text
An architecture that makes attention, especially self-attention, the central sequence-processing mechanism.
```

---

# 59. Important Equations

## Dot Product Score

```text
score(q, k) = q^T k
```

---

## General Attention

```text
score(q, k) = q^T W k
```

---

## Additive Attention

```text
score(q, k)
=
v^T tanh(Wq q + Wk k + b)
```

---

## Attention Weight

```text
alpha_i
=
softmax(score_i)
```

---

## Context Vector

```text
context
=
sum(alpha_i * value_i)
```

---

## Scaled Dot Product Attention

```text
Attention(Q, K, V)
=
softmax(
    (Q K^T) / sqrt(d_k)
) V
```

---

# 60. Interview-Level Definition

Attention is a mechanism that allows a neural network to dynamically assign different importance weights to available representations based on their relevance to the current query.

In classical Seq2Seq models, attention allows the decoder to directly access all encoder hidden states instead of relying only on the encoder's final fixed-size context representation.

In modern architectures, attention is generalized using Query, Key, and Value representations.

---

# 61. Interview-Level Explanation of Q, K, V

```text
Query
=
What information am I looking for?

Key
=
Does this position contain relevant information?

Value
=
What information should I retrieve from this position?
```

The Query is compared with Keys.

The comparison produces attention scores.

Softmax converts scores into weights.

The weights are applied to Values.

---

# 62. Complete Attention Flow

```text
Available Representations
        |
        v
       Keys
       Values

Current Representation
        |
        v
       Query

Query compared with Keys
        |
        v
Attention Scores
        |
        v
Softmax
        |
        v
Attention Weights
        |
        v
Weighted Combination of Values
        |
        v
Contextual Representation
```

---

# 63. Complete Evolution Toward Transformers

```text
Bag of Words / TF-IDF
    ->
Static Word Embeddings
    ->
RNN
    ->
LSTM / GRU
    ->
Seq2Seq Encoder-Decoder
    ->
Seq2Seq + Attention
    ->
Self-Attention
    ->
Scaled Dot Product Attention
    ->
Multi-Head Attention
    ->
Transformer
```

Each stage addressed a different limitation.

```text
BoW / TF-IDF:
Lost semantic relationships and sequence order.

Word Embeddings:
Added semantic representations but remained static.

RNN:
Added sequential context.

LSTM / GRU:
Improved long-term dependency handling.

Seq2Seq:
Enabled variable-length sequence transformation.

Attention:
Removed the fixed-context bottleneck.

Self-Attention:
Allowed tokens to directly model relationships with other tokens.

Transformer:
Removed recurrence as the central mechanism and built the architecture around attention.
```

---

# 64. Final Mental Model

The simplest way to remember attention is:

```text
Attention does three main things:

1. Ask what information is needed.
2. Find where that information is located.
3. Retrieve a weighted combination of that information.
```

Using Q, K, and V:

```text
Query
=
What do I need?

Key
=
Where should I look?

Value
=
What information should I take?
```

Mathematically:

```text
Query + Keys
    ->
Scores
    ->
Softmax
    ->
Weights

Weights + Values
    ->
Context
```

This mechanism began as a solution to the fixed-context problem in Seq2Seq models and eventually became one of the core ideas behind the Transformer architecture.