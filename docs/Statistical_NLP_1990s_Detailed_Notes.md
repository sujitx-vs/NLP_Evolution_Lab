# Statistical NLP (1990s) - Detailed Study Notes

## Introduction

Statistical NLP marked the transition from manually written linguistic
rules to learning patterns from data. Instead of asking **"What grammar
rule should I write?"**, statistical NLP asks **"What does the data tell
us?"**

The overall pipeline became:

``` text
Raw Text
   │
Preprocessing
(lowercase, tokenize, clean)
   │
Feature Extraction
(BoW, TF-IDF, N-grams)
   │
Statistical Learning
(HMM, SVM)
   │
Prediction
```

------------------------------------------------------------------------

# 1. Bag of Words (BoW)

## Concept

BoW converts text into numbers by counting how many times each
vocabulary word appears.

Example:

Sentence A:

    I love NLP

Sentence B:

    I love Python

Vocabulary

    [I, love, NLP, Python]

Vectors

    A = [1,1,1,0]
    B = [1,1,0,1]

## Characteristics

-   Word order is ignored.
-   Every word becomes one feature.
-   Vector size equals vocabulary size.

### Advantages

-   Extremely simple.
-   Fast.
-   Good baseline for text classification.

### Limitations

-   Cannot understand context.
-   "Dog bites man" and "Man bites dog" look similar.
-   Produces sparse, high-dimensional vectors.

------------------------------------------------------------------------

# 2. TF-IDF

BoW treats every word equally. TF-IDF assigns importance.

## Term Frequency (TF)

Measures how important a word is inside one document.

## Inverse Document Frequency (IDF)

Measures how rare a word is across the whole corpus.

Words like "the", "is", "of" receive low scores.

Rare informative words receive higher scores.

Overall:

TF-IDF = TF × IDF

### Why it was introduced

To reduce the influence of extremely common words.

### Applications

-   Search engines
-   Document ranking
-   Text classification
-   Information retrieval

------------------------------------------------------------------------

# 3. N-grams

An N-gram is a sequence of N consecutive words.

Example:

    I love natural language processing

Unigrams

    I
    love
    natural
    language
    processing

Bigrams

    I love
    love natural
    natural language
    language processing

Trigrams

    I love natural
    love natural language
    natural language processing

### Why use N-grams?

BoW loses word order.

N-grams preserve local context.

Example:

    machine learning

contains more information than

    machine
    learning

Applications: - Next-word prediction - Spell correction - Autocomplete -
Language modelling

Limitations: - Data sparsity - Exploding vocabulary - Cannot remember
long-distance relationships

------------------------------------------------------------------------

# 4. Markov Model

## Assumption

The next word depends only on the previous state(s).

Example

    I -> love
    love -> NLP

Training:

Corpus → Count transitions → Compute probabilities

Prediction:

Current word → Find all possible next words → Choose highest probability

Strength: - Simple probabilistic language model.

Weakness: - Very limited memory. - No semantic understanding.

------------------------------------------------------------------------

# 5. Hidden Markov Model (HMM)

HMM introduces hidden states.

Observed states: Words

Hidden states: POS tags

Example

    Words:
    Dogs bark loudly

    Hidden:
    NNS VBP RB

The model learns three probabilities.

## Initial Probability

Probability of starting with a tag.

## Transition Probability

    P(Tag2 | Tag1)

Probability that one tag follows another.

Example

    DT -> NN
    NN -> VBZ

## Emission Probability

    P(Word | Tag)

Probability that a hidden state emits a word.

Example

    NN emits "dog"
    VB emits "run"

Applications: - POS tagging - Historical speech recognition - Early NER

Limitations: - Strong independence assumptions. - Short context.

------------------------------------------------------------------------

# 6. Support Vector Machine (SVM)

SVM is a supervised classification algorithm.

Workflow

    Text
     ↓
    TF-IDF
     ↓
    Feature Vector
     ↓
    Every document becomes a point
     ↓
    Find separating hyperplane
     ↓
    Choose maximum margin
     ↓
    Predict class

## Hyperplane

The separator between classes.

Decision function

wᵀx + b = 0

Prediction

Positive side → Positive class

Negative side → Negative class

## Margin

Distance between the hyperplane and the closest training points.

The larger the margin, the better the model usually generalizes.

## Support Vectors

Only the closest points determine the hyperplane.

Removing far-away points usually changes nothing.

Removing support vectors changes the boundary.

Applications: - Spam detection - Sentiment analysis - Document
classification - Topic classification

Advantages - Excellent for sparse text. - High accuracy on small/medium
datasets. - Fast inference.

Limitations - Needs feature engineering. - No semantic understanding.

------------------------------------------------------------------------

# Summary

  ------------------------------------------------------------------------
  Technique        Main Idea        Typical Use         Limitation
  ---------------- ---------------- ------------------- ------------------
  BoW              Count words      Classification      No context

  TF-IDF           Weight important Search, Retrieval   No semantics
                   words                                

  N-grams          Preserve local   Language modelling  Short memory
                   order                                

  Markov           Predict next     Autocomplete        Very limited
                   state                                context

  HMM              Hidden state     POS Tagging         Independence
                   modelling                            assumptions

  SVM              Maximum-margin   Sentiment, Spam     Needs engineered
                   classifier                           features
  ------------------------------------------------------------------------

------------------------------------------------------------------------

# Why Statistical NLP Was Eventually Replaced

Although statistical methods were revolutionary, they suffered from:

-   Sparse vectors
-   Manual feature engineering
-   No semantic understanding
-   No long-range context
-   Difficulty scaling to complex language

These limitations led to the Deep Learning era:

    Bag of Words
          ↓
    TF-IDF
          ↓
    N-grams
          ↓
    HMM / SVM
          ↓
    Word2Vec
          ↓
    GloVe
          ↓
    RNN
          ↓
    LSTM
          ↓
    Seq2Seq
