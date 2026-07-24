import nltk
import re

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

file_path = "datasets/txt_sample.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("=" * 60)
print("STEP 0 : ORIGINAL TEXT")
print("=" * 60)
print(text)


# =====================================================
# TOKENIZE EACH SENTENCE
# =====================================================

print("\n" + "=" * 60)
print("TOKENS PER SENTENCE")
print("=" * 60)

nltk_sentences = sent_tokenize(text)

for i, sentence in enumerate(nltk_sentences, start = 1):
    print(f"Sentences {i} : {sentence}")


# =====================================================
# WORD TOKENIZER
# =====================================================


print("\n" + "=" * 60)
print("NLTK WORD TOKENIZER")
print("=" * 60)

tokenized_text = []

for i, sentence in enumerate(nltk_sentences, start=1):
    tokens = word_tokenize(sentence)
    tokenized_text.append(tokens)

print(tokenized_text)

# =====================================================
# LOWERCASING
# =====================================================

print("\n" + "=" * 60)
print("LOWERCASING")
print("=" * 60)

lowercase_document = []

for i, tokens in enumerate(tokenized_text, start=1):

    lowercase_tokens = []

    for token in tokens:
        lowercase_tokens.append(token.lower())

    lowercase_document.append(lowercase_tokens)

    print(f"\nSentence {i}")
    print(lowercase_tokens)

# =====================================================
# REMOVE PUNCTUATION
# =====================================================

print("\n" + "=" * 60)
print("REMOVE PUNCTUATION")
print("=" * 60)

clean_document = []

pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*|\d+(?:\.\d+)?"

for i, tokens in enumerate(lowercase_document, start=1):

    cleaned_tokens = []

    for token in tokens:

        if re.fullmatch(pattern, token):
            cleaned_tokens.append(token)

    clean_document.append(cleaned_tokens)

    print(f"\nSentence {i}")
    print(cleaned_tokens)



# =====================================================
# STOPWORD REMOVAL
# =====================================================

print("\n" + "=" * 60)
print("STOPWORD REMOVAL")
print("=" * 60)

filtered_document = []

for i, tokens in enumerate(clean_document, start=1):

    filtered_tokens = []

    for token in tokens:

        if token not in stop_words:
            filtered_tokens.append(token)

    filtered_document.append(filtered_tokens)

    print(f"\nSentence {i}")
    print(filtered_tokens)

# =====================================================
# LEMMATIZATION
# =====================================================

print("\n" + "=" * 60)
print("LEMMATIZATION")
print("=" * 60)

lemmatized_document = []

for i, tokens in enumerate(filtered_document, start=1):

    lemmatized_tokens = []

    for token in tokens:
        lemmatized_tokens.append(
            lemmatizer.lemmatize(token)
        )

    lemmatized_document.append(lemmatized_tokens)

    print(f"\nSentence {i}")
    print(lemmatized_tokens)


# =====================================================
# BUILD VOCABULARY
# =====================================================

print("\n" + "=" * 60)
print("BUILD VOCABULARY")
print("=" * 60)

vocabulary = sorted(set(
    word
    for sentence in lemmatized_document
    for word in sentence
))

print(vocabulary)

print("\nVocabulary Size :", len(vocabulary))


# =====================================================
# STEP 7 : BAG OF WORDS (FROM SCRATCH)
# =====================================================

print("\n" + "=" * 60)
print("BAG OF WORDS (FROM SCRATCH)")
print("=" * 60)

bow_matrix = []

for sentence in lemmatized_document:

    # Create one row filled with zeros
    bow_vector = [0] * len(vocabulary)

    # Count each word
    for word in sentence:

        if word in vocabulary:

            index = vocabulary.index(word)

            bow_vector[index] += 1

    bow_matrix.append(bow_vector)


print("\nVocabulary")
print(vocabulary)

print("\nBag of Words Matrix\n")

for i, vector in enumerate(bow_matrix, start=1):
    print(f"Sentence {i}")
    print(vector)


# =====================================================
# WORD TO INDEX MAP
# =====================================================

print("\n" + "=" * 60)
print("WORD TO INDEX MAP")
print("=" * 60)

word_to_index = {}

for index, word in enumerate(vocabulary):
    word_to_index[word] = index

print(word_to_index)



# =====================================================
# STEP 9 : FAST BAG OF WORDS
# =====================================================

print("\n" + "=" * 60)
print("FAST BAG OF WORDS")
print("=" * 60)

fast_bow = []

for sentence in lemmatized_document:

    vector = [0] * len(vocabulary)

    for word in sentence:

        if word in word_to_index:

            index = word_to_index[word]

            vector[index] += 1

    fast_bow.append(vector)

for i, vector in enumerate(fast_bow, start=1):
    print(f"\nSentence {i}")
    print(vector)


# =====================================================
# COUNT VECTORIZATION (SCIKIT-LEARN)
# =====================================================

print("\n" + "=" * 60)
print("COUNT VECTORIZATION (SCIKIT-LEARN)")
print("=" * 60)

# Convert token lists back into sentences
processed_sentences = []

for sentence in lemmatized_document:
    processed_sentences.append(" ".join(sentence))

# Build Count Vectorizer
vectorizer = CountVectorizer()

count_matrix = vectorizer.fit_transform(processed_sentences)

# Vocabulary
print("\nVocabulary")
print(vectorizer.get_feature_names_out())

# Matrix
print("\nCount Matrix\n")
print(count_matrix.toarray())



# =====================================================
# DOCUMENT FREQUENCY (DF)
# =====================================================

import math

print("\n" + "=" * 60)
print("DOCUMENT FREQUENCY (DF)")
print("=" * 60)

document_frequency = {}

for word in vocabulary:

    count = 0

    for sentence in lemmatized_document:

        if word in sentence:
            count += 1

    document_frequency[word] = count

for word, df in document_frequency.items():
    print(f"{word:15} --> {df}")



# =====================================================
# INVERSE DOCUMENT FREQUENCY (IDF)
# =====================================================

print("\n" + "=" * 60)
print("INVERSE DOCUMENT FREQUENCY (IDF)")
print("=" * 60)

number_of_documents = len(lemmatized_document)

idf = {}

for word, df in document_frequency.items():

    idf[word] = math.log(number_of_documents / df)

for word, score in idf.items():
    print(f"{word:15} --> {score:.4f}")


# =====================================================
# TERM FREQUENCY (TF)
# =====================================================

print("\n" + "=" * 60)
print("TERM FREQUENCY (TF)")
print("=" * 60)

tf_document = []

for i, sentence in enumerate(lemmatized_document, start=1):

    tf = {}

    total_words = len(sentence)

    for word in vocabulary:

        count = sentence.count(word)

        tf[word] = count / total_words if total_words > 0 else 0

    tf_document.append(tf)

    print(f"\nSentence {i}")

    for word, score in tf.items():

        if score > 0:
            print(f"{word:15} --> {score:.4f}")



# =====================================================
# TF-IDF
# =====================================================

print("\n" + "=" * 60)
print("TF-IDF")
print("=" * 60)

tfidf_document = []

for i, tf in enumerate(tf_document, start=1):

    tfidf = {}

    for word in vocabulary:

        tfidf[word] = tf[word] * idf[word]

    tfidf_document.append(tfidf)

    print(f"\nSentence {i}")

    for word, score in tfidf.items():

        if score > 0:
            print(f"{word:15} --> {score:.4f}")


# =====================================================
# TF-IDF MATRIX
# =====================================================

print("\n" + "=" * 60)
print("TF-IDF MATRIX")
print("=" * 60)

print("\nVocabulary")
print(vocabulary)

print("\nMatrix")

for i, tfidf in enumerate(tfidf_document, start=1):

    vector = []

    for word in vocabulary:
        vector.append(round(tfidf[word], 4))

    print(f"\nSentence {i}")
    print(vector)