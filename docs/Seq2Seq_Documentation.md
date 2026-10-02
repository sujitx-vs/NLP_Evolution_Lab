# Seq2Seq (Sequence-to-Sequence): A Deeper Understanding

## 1. What is Seq2Seq?

Seq2Seq is a deep learning architecture that maps a **variable-length input sequence** to a **variable-length output sequence**, where the two lengths need not be equal.

$$
X = (x_1, x_2, \dots, x_T) \;\longrightarrow\; Y = (y_1, y_2, \dots, y_{T'}), \qquad T \neq T' \text{ in general}
$$

Typical uses: machine translation, summarization, chatbots, grammar correction, question generation.

---

## 2. Why was it introduced?

Plain RNNs, LSTMs, and GRUs fit these task shapes well:

| Task type | Shape | Example |
|---|---|---|
| Many-to-one | $T$ inputs → 1 output | Sentiment classification |
| Many-to-many (aligned) | $T$ inputs → $T$ outputs | POS tagging, NER |
| **Many-to-many (unaligned)** | $T$ inputs → $T'$ outputs | **Translation, summarization** |

The third case is the problem. In translation, the output length differs from the input length and word order can change:

```
Input  (3 tokens): I love NLP .
Output (4 tokens): J' aime le NLP .
```

A single RNN emits one output per input step, so it cannot handle this. Seq2Seq solves it by **separating reading from writing**.

---

## 3. Main Innovation: Encoder–Decoder Architecture

```
Input Sentence
      ↓
   Encoder  (RNN / LSTM / GRU)
      ↓
Final Hidden State  h_T   ← the "Context Vector"
      ↓
   Decoder  (RNN / LSTM / GRU)
      ↓
Output Sentence
```

Two separate recurrent networks, with separate parameters:

- **Encoder** reads the input and compresses it into a vector.
- **Decoder** is a conditional language model: it generates the output one token at a time, conditioned on that vector.

---

## 4. Encoder

### 4.1 Embedding

Each input token index $x_t$ is mapped to a dense vector using an embedding matrix $E_{enc}$:

$$
e_t = E_{enc}\, x_t \qquad e_t \in \mathbb{R}^{d_{emb}}
$$

### 4.2 Recurrence

The encoder is a recurrent network that updates its hidden state at every time step:

$$
h_t = f_{enc}(h_{t-1},\, e_t), \qquad h_0 = \mathbf{0}
$$

For a vanilla RNN:

$$
h_t = \tanh(W_{hh}\, h_{t-1} + W_{xh}\, e_t + b_h)
$$

For a GRU or LSTM, $f_{enc}$ is the gated version (see section 4.3).

Each $h_t$ summarizes everything the encoder has read **up to** position $t$.

### 4.3 LSTM version

If the encoder is an LSTM, it carries two states: a hidden state $h_t$ and a cell state $c_t$.

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, e_t] + b_f) &&\text{(forget gate)}\\
i_t &= \sigma(W_i [h_{t-1}, e_t] + b_i) &&\text{(input gate)}\\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, e_t] + b_c) &&\text{(candidate cell)}\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t &&\text{(cell update)}\\
o_t &= \sigma(W_o [h_{t-1}, e_t] + b_o) &&\text{(output gate)}\\
h_t &= o_t \odot \tanh(c_t) &&\text{(hidden state)}
\end{aligned}
$$

---

## 5. Context Vector and the Final Hidden State Handoff

This is the core of the original (native, "first-generation") Seq2Seq model.

After the encoder has consumed all $T$ tokens, the **final hidden state** is taken as the context vector:

$$
\boxed{\,c = h_T\,}
$$

Everything the decoder will ever know about the input must be inside this one vector.

### 5.1 How it is passed to the decoder

The decoder's initial hidden state is **initialized with the encoder's final hidden state**:

$$
s_0 = h_T
$$

where $s_t$ denotes the decoder hidden state.

For an **LSTM**, both states are transferred:

$$
s_0 = h_T, \qquad c^{dec}_0 = c^{enc}_T
$$

Notes:

- The encoder and decoder must have the **same hidden size** for a direct copy (otherwise a linear projection $s_0 = W_c h_T + b_c$ is used).
- If the encoder has multiple layers, layer $l$ of the encoder passes its final state to layer $l$ of the decoder.
- Intermediate states $h_1, \dots, h_{T-1}$ are **discarded** in native Seq2Seq. Only $h_T$ survives.

### 5.2 Two variants in the original papers

| Variant | Paper | How $c$ is used |
|---|---|---|
| Initial-state only | Sutskever et al., 2014 | $s_0 = c$; decoder never sees $c$ again explicitly |
| Context at every step | Cho et al., 2014 | $c$ is fed as an extra input at **every** decoder step |

Cho-style update:

$$
s_t = f_{dec}(s_{t-1},\, [\,y_{t-1};\, c\,])
$$

Both variants still rely on a single fixed-length vector.

### 5.3 Visual flow

```
x1 → [Enc] → h1
x2 → [Enc] → h2
x3 → [Enc] → h3 = c  ──────────────┐
                                   ↓  (s0 = h3)
<SOS> → [Dec] → s1 → y1 ("J'")
 y1   → [Dec] → s2 → y2 ("aime")
 y2   → [Dec] → s3 → y3 ("le")
 ...                      until <EOS>
```

---

## 6. Decoder

The decoder is an autoregressive generator. It starts from $s_0 = c$ and produces tokens until it emits an end token.

### 6.1 Step-by-step equations

At each step $t = 1, 2, \dots$:

1. Embed the **previous** output token (initially the `<SOS>` token $y_0$):

$$
d_t = E_{dec}\, y_{t-1}
$$

2. Update the decoder hidden state:

$$
s_t = f_{dec}(s_{t-1},\, d_t)
$$

3. Project to vocabulary size $|V|$ and apply softmax:

$$
o_t = W_o\, s_t + b_o, \qquad
P(y_t \mid y_{<t}, X) = \text{softmax}(o_t)
$$

4. Choose the next token (see section 8) and repeat until `<EOS>`.

### 6.2 Probabilistic view

Seq2Seq models the conditional probability of the whole output sequence, factorized by the chain rule:

$$
P(Y \mid X) = \prod_{t=1}^{T'} P\big(y_t \mid y_1, \dots, y_{t-1},\, c\big)
$$

The dependence on $X$ passes **entirely** through $c = h_T$. That is the bottleneck, discussed in section 10.

---

## 7. Training

### 7.1 Loss function

Training maximizes the log-likelihood of the correct output sequence, which is equivalent to minimizing cross-entropy:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{T'} \log P\big(y_t^{*} \mid y_{<t}^{*},\, X;\, \theta\big)
$$

where $y^*$ is the ground-truth token. Over a dataset of $N$ pairs:

$$
\mathcal{L}_{total} = -\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T'_n} \log P\big(y^{*(n)}_t \mid y^{*(n)}_{<t},\, X^{(n)}\big)
$$

### 7.2 Teacher forcing

During training, the decoder input at step $t$ is the **true** previous token $y^*_{t-1}$, not the model's own prediction:

$$
d_t = E_{dec}\, y^{*}_{t-1}
$$

This speeds up and stabilizes training, but it creates **exposure bias**: at inference time the model must consume its own (possibly wrong) predictions, which it never practiced on.

### 7.3 Backpropagation

Gradients flow from the loss, through the decoder, **across the context vector** $h_T$, and back through the encoder (BPTT):

$$
\frac{\partial \mathcal{L}}{\partial \theta_{enc}}
= \frac{\partial \mathcal{L}}{\partial s_0}\cdot\frac{\partial s_0}{\partial h_T}\cdot\frac{\partial h_T}{\partial \theta_{enc}}
$$

Because every gradient to the encoder must pass through this one vector, and then through $T$ recurrent steps, vanishing gradients are a real concern. Gated units (LSTM/GRU) and gradient clipping are standard.

---

## 8. Inference (Generation)

No ground truth is available, so the decoder feeds on its own outputs.

**Greedy decoding:** pick the most probable token at each step.

$$
\hat{y}_t = \arg\max_{w \in V} P(w \mid \hat{y}_{<t}, c)
$$

**Beam search:** keep the top-$k$ partial sequences (the beam) by cumulative log-probability:

$$
\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, c)
$$

Beam search usually yields better translations than greedy decoding, since a locally best token may lead to a poor sentence overall. A length normalization term $\frac{1}{t^{\alpha}}$ is commonly applied so that longer sequences are not unfairly penalized.

Generation stops when `<EOS>` is produced or a maximum length is reached.

---

## 9. Full Workflow

1. **Tokenize** the input (and add `<SOS>` / `<EOS>` markers on the target side).
2. **Embed** each token: $e_t = E_{enc}x_t$.
3. **Encode**: $h_t = f_{enc}(h_{t-1}, e_t)$ for $t = 1..T$.
4. **Extract context**: $c = h_T$ (plus $c^{enc}_T$ for LSTMs).
5. **Initialize decoder**: $s_0 = c$.
6. **Decode** step by step: $s_t = f_{dec}(s_{t-1}, d_t)$, then $P(y_t) = \text{softmax}(W_o s_t + b_o)$.
7. **Stop** at `<EOS>`.

### Tiny worked example

Input: `I love NLP` → Output: `J'aime le NLP`

```
Encoder:  h0=0 → h1(I) → h2(love) → h3(NLP) = c
Decoder:  s0=c
          <SOS> → s1 → "J'aime"
          "J'aime" → s2 → "le"
          "le" → s3 → "NLP"
          "NLP" → s4 → <EOS>
```

---

## 10. Shortcomings: The Context Vector Bottleneck

Whether the input has 5 words or 80 words, it must be squeezed into one vector $h_T \in \mathbb{R}^{d}$ of fixed size $d$.

Why this hurts:

1. **Fixed capacity.** The information to store grows with $T$, but the vector size stays at $d$. Information loss is unavoidable for long sentences.
2. **Recency bias.** $h_T$ is built last, so it is dominated by the final tokens. Early tokens fade through repeated recurrent updates.
3. **Long gradient path.** The influence of $x_1$ on the output must travel through $T$ steps; the gradient shrinks roughly like

$$
\left\| \frac{\partial h_T}{\partial h_1} \right\| \sim \prod_{t=2}^{T} \left\| \frac{\partial h_t}{\partial h_{t-1}} \right\|
$$

   which vanishes (or explodes) when the factors are consistently below (or above) 1.
4. **No selective access.** When generating word $y_t$, the decoder cannot look back at the specific input words relevant to it.

Empirically, translation quality (BLEU) drops sharply as sentence length grows (Cho et al., 2014).

### Early workarounds

- **Reverse the input sequence** (Sutskever et al., 2014): the first source words end up closer to the first target words, shortening the dependency path.
- **Deeper / wider LSTMs** and **bidirectional encoders**.

These help but do not remove the bottleneck.

---

## 11. What came next? Attention

Instead of using only $h_T$, keep **all** encoder states $h_1, \dots, h_T$ and let the decoder build a **different context vector for every output step**.

**Alignment score** between decoder state $s_{t-1}$ and each encoder state $h_i$:

$$
e_{t,i} = \text{score}(s_{t-1}, h_i)
\quad\text{e.g. (Bahdanau)}\quad
e_{t,i} = v^\top \tanh(W_s s_{t-1} + W_h h_i)
$$

**Attention weights** (a distribution over input positions):

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T}\exp(e_{t,j})}
$$

**Dynamic context vector**:

$$
c_t = \sum_{i=1}^{T} \alpha_{t,i}\, h_i
$$

**Decoder update** now uses $c_t$ instead of a single fixed $c$:

$$
s_t = f_{dec}(s_{t-1},\, [\,y_{t-1};\, c_t\,])
$$

| | Native Seq2Seq | Seq2Seq + Attention |
|---|---|---|
| Context | one fixed $c = h_T$ | $c_t$ recomputed every step |
| Encoder states used | only $h_T$ | all $h_1..h_T$ |
| Long sentences | degrades | much better |
| Interpretability | none | alignment weights $\alpha_{t,i}$ |

---

## 12. Applications

- Machine Translation
- Text Summarization
- Chatbots / Dialogue Systems
- Grammar Correction
- Question Generation
- (Also) Speech recognition, image captioning (CNN encoder + RNN decoder), code generation

## 13. Advantages

- Handles variable-length inputs and outputs.
- End-to-end differentiable; no hand-built alignment or phrase tables.
- Foundation of Neural Machine Translation.
- Generic: encoder and decoder can be swapped for other architectures.

## 14. Limitations (Summary)

- Fixed-size context vector bottleneck.
- Poor long-range memory.
- Exposure bias from teacher forcing.
- Sequential computation, so no parallelism across time steps (slow training).

---

## 15. Historical Position

```
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
Seq2Seq          ← fixed context vector c = h_T
   ↓
Attention        ← dynamic context c_t
   ↓
Transformer      ← attention only, no recurrence
   ↓
BERT / GPT
```

## 16. Key References

- Sutskever, Vinyals, Le (2014). *Sequence to Sequence Learning with Neural Networks.*
- Cho et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.*
- Bahdanau, Cho, Bengio (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.*
- Vaswani et al. (2017). *Attention Is All You Need.*