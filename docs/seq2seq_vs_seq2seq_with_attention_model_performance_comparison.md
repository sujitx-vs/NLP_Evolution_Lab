# Model Performance Comparison: Standard Seq2Seq vs. Seq2Seq with Attention

## Executive Summary

Adding an **Attention Mechanism** to the baseline Sequence-to-Sequence (Seq2Seq) model significantly improves performance across both evaluation metrics on the **Manglish** dataset. 

The attention-equipped model achieves an **absolute increase of +7.83%** in Exact Match Accuracy and an **absolute increase of +0.1174** (+16.04% relative gain) in Corpus BLEU.

---

## Metric Comparison Table

| Metric | Baseline (Standard Seq2Seq) | Seq2Seq with Attention | Absolute Difference | Relative Gain (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Match Accuracy** | `0.6442` (64.42%) | **`0.7225` (72.25%)** | `+0.0783` (+7.83 pts) | **+12.16%** |
| **Corpus BLEU** | `0.7316` | **`0.8490`** | `+0.1174` | **+16.04%** |

---

## Metric Breakdown & Analysis

### 1. Exact Match Accuracy (`0.6442` $\rightarrow$ `0.7225`)
* **Definition:** Measures the percentage of generated sequence outputs that perfectly match the target reference sequence word-for-word and character-for-character.
* **Analysis:**
  * The baseline model suffers from the classic "bottleneck problem"—forcing the entire input sequence into a single, fixed-sized context vector.
  * Seq2Seq with Attention allows the decoder to selectively look back at specific encoder states at each decoding step, preventing context loss and boosting exact sequence reconstruction by **+7.83 percentage points**.

### 2. Corpus BLEU (`0.7316` $\rightarrow$ `0.8490`)
* **Definition:** Measures $n$-gram overlap and structural similarity across the evaluation dataset while penalizing length discrepancies.
* **Analysis:**
  * Baseline Seq2Seq gets a reasonable `0.7316`, proving it captures general linguistic structure, but frequently misses or swaps specific target tokens.
  * The Attention variant achieves **`0.8490`** (**+16.04% relative gain**), demonstrating superior word-alignment, local fluency, and token ordering.

---

## Key Conclusions & Recommendation

1. **Information Bottleneck Elimination:** Attention removes the need to compress long source inputs into a single vector.
2. **Superior Alignment:** For transliteration/translation tasks in Manglish, explicit soft-alignment learned by Attention significantly improves target token mapping.
3. **Recommendation:** **Seq2Seq with Attention** is strictly superior across all metrics and should be selected for downstream deployment.