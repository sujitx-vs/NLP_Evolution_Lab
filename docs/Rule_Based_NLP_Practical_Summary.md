# Rule-Based NLP - Practical Learning Summary

This document summarizes the classical Rule-Based NLP pipeline explored through the practical exercises.

## Classical NLP Pipeline

```
Raw Text
 -> Text Reading
 -> Whitespace Normalization
 -> Sentence Segmentation
 -> Word Tokenization
 -> Text Cleaning
    -> Rule-Based Cleaning
    -> Regex-Based Cleaning
 -> Stopword Removal
 -> Stemming
 -> Lemmatization
 -> POS Tagging
 -> Named Entity Recognition (NER)
```

## Summary

### 1. Text Reading
- **Why:** Load text into memory.
- **Problem Solved:** Converts file contents into a Python string.
- **How:** `open()` and `read()`.
- **Limitations:** Only reads characters.

### 2. Whitespace Normalization
- **Why:** Standardize formatting.
- **Problem Solved:** Removes extra spaces, tabs and newlines.
- **How:** `re.sub(r"\\s+", " ", text)`.
- **Limitations:** Doesn't understand language.

### 3. Sentence Segmentation
- **Why:** Identify sentence boundaries.
- **Problem Solved:** Splits documents into sentences.
- **How:** `split('.')` or `sent_tokenize()`.
- **Limitations:** Naive splitting fails for emails, decimals and abbreviations.

### 4. Word Tokenization
- **Why:** Split sentences into words.
- **Problem Solved:** Produces tokens.
- **How:** `split()` or `word_tokenize()`.
- **Limitations:** Difficult cases include contractions, emails and abbreviations.

### 5. Text Cleaning
- **Why:** Remove unwanted characters.
- **Problem Solved:** Cleaner tokens.
- **How:** `strip()` or regex.
- **Limitations:** Regex may remove meaningful symbols.

### 6. Stopword Removal
- **Why:** Remove common function words.
- **Problem Solved:** Reduces vocabulary.
- **How:** Compare with stopword list.
- **Limitations:** May remove meaningful words like 'not'.

### 7. Stemming
- **Why:** Reduce related words to one stem.
- **Problem Solved:** Vocabulary reduction.
- **How:** Rule-based suffix stripping.
- **Limitations:** May produce non-dictionary words.

### 8. Lemmatization
- **Why:** Produce dictionary forms.
- **Problem Solved:** Better normalization.
- **How:** Dictionary + linguistic rules.
- **Limitations:** Without POS, results may be incorrect.

### 9. POS Tagging
- **Why:** Identify grammatical role.
- **Problem Solved:** Assign noun, verb, adjective, etc.
- **How:** Context-aware tagging.
- **Limitations:** Tags may be incorrect.

### 10. Lemmatization with POS
- **Why:** Improve lemma accuracy.
- **Problem Solved:** Uses grammar for correct base form.
- **How:** POS mapping before lemmatization.
- **Limitations:** Depends on POS accuracy.

### 11. Named Entity Recognition (NER)
- **Why:** Detect real-world entities.
- **Problem Solved:** Finds people, organizations and locations.
- **How:** Context-based sequence labeling.
- **Limitations:** May misclassify entities.

## Key Takeaways

The Rule-Based NLP pipeline gradually transforms raw text into structured linguistic information through a series of deterministic processing stages.