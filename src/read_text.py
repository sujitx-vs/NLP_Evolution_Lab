import re
import nltk

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk import ne_chunk

nltk.download("punkt_tab")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
nltk.download("wordnet")
nltk.download("omw-1.4")
lemmatizer = WordNetLemmatizer()
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")

# =====================================================
# STEP 0: READ THE TEXT FILE
# =====================================================

file_path = "datasets/txt_sample01.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("=" * 60)
print("STEP 0 : ORIGINAL TEXT")
print("=" * 60)
print(text)

# =====================================================
# STEP 1: WHITESPACE NORMALIZATION
# =====================================================

normalized_text = re.sub(r"\s+", " ", text).strip()

print("\n" + "=" * 60)
print("STEP 1 : NORMALIZED TEXT")
print("=" * 60)
print(normalized_text)

# =====================================================
# STEP 2: NAIVE SENTENCE SEGMENTATION
# =====================================================

raw_sentences = normalized_text.split(".")

# Remove empty sentences
sentences = [s.strip() for s in raw_sentences if s.strip()]

print("\n" + "=" * 60)
print("STEP 2 : SENTENCES")
print("=" * 60)

for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence}")

# =====================================================
# STEP 3: TOKENIZE EACH SENTENCE
# =====================================================

document_tokens = []

print("\n" + "=" * 60)
print("STEP 3 : TOKENS PER SENTENCE")
print("=" * 60)

for i, sentence in enumerate(sentences, start=1):

    tokens = sentence.split()

    document_tokens.append(tokens)

    print(f"\nSentence {i}")
    print(tokens)

# =====================================================
# STEP 4: RULE-BASED CLEANING
# =====================================================

cleaned_document = []

print("\n" + "=" * 60)
print("STEP 4 : RULE-BASED CLEANING")
print("=" * 60)

for i, tokens in enumerate(document_tokens, start=1):

    cleaned_tokens = []

    for token in tokens:
        cleaned_tokens.append(token.strip(".,!?$"))

    cleaned_document.append(cleaned_tokens)

    print(f"\nSentence {i}")
    print(cleaned_tokens)

# =====================================================
# STEP 5: REGEX-BASED CLEANING
# =====================================================

regex_document = []

print("\n" + "=" * 60)
print("STEP 5 : REGEX-BASED CLEANING")
print("=" * 60)

for i, tokens in enumerate(document_tokens, start=1):

    regex_tokens = []

    for token in tokens:
        cleaned = re.sub(r"[^\w\s]", "", token)
        regex_tokens.append(cleaned)

    regex_document.append(regex_tokens)

    print(f"\nSentence {i}")
    print(regex_tokens)



# =====================================================
# STEP 6 : NLTK SENTENCE TOKENIZER
# =====================================================

nltk_sentences = sent_tokenize(text)

print("\n" + "=" * 60)
print("STEP 6 : NLTK SENTENCE TOKENIZER")
print("=" * 60)

for i, sentence in enumerate(nltk_sentences, start=1):
    print(f"Sentence {i}: {sentence}")

# =====================================================
# STEP 7 : NLTK WORD TOKENIZER
# =====================================================


print("\n" + "=" * 60)
print("STEP 7 : NLTK WORD TOKENIZER")
print("=" * 60)

for i, sentence in enumerate(nltk_sentences, start=1):
    tokens = word_tokenize(sentence)

    print(f"\nSentence {i}")
    print(tokens)

# =====================================================
# STEP 8 : STOPWORD REMOVAL
# =====================================================

print("\n" + "=" * 60)
print("STEP 8 : STOPWORD REMOVAL")
print("=" * 60)
 
for i, sentence in enumerate(nltk_sentences, start=1):

    tokens = word_tokenize(sentence)

    filtered_tokens = []

    for token in tokens:

        # Ignore punctuation and compare in lowercase
        if token.lower() not in stop_words and token.isalpha():
            filtered_tokens.append(token)

    print(f"\nSentence {i}")
    print(filtered_tokens)


# =====================================================
# STEP 9 : PORTER STEMMING
# =====================================================

print("\n" + "=" * 60)
print("STEP 9 : PORTER STEMMING")
print("=" * 60)

for i, sentence in enumerate(nltk_sentences, start=1):

    tokens = word_tokenize(sentence)

    stemmed_tokens = []

    for token in tokens:

        if token.isalpha():
            stemmed_tokens.append(stemmer.stem(token))

    print(f"\nSentence {i}")
    print(stemmed_tokens)



# =====================================================
# STEP 10 : LEMMATIZATION (WITHOUT POS)
# =====================================================

print("\n" + "=" * 60)
print("STEP 10 : LEMMATIZATION (WITHOUT POS)")
print("=" * 60)

for i, sentence in enumerate(nltk_sentences, start=1):

    tokens = word_tokenize(sentence)

    lemmatized_tokens = []

    for token in tokens:

        # Ignore punctuation
        if token.isalpha():
            lemmatized_tokens.append(lemmatizer.lemmatize(token.lower()))

    print(f"\nSentence {i}")
    print(lemmatized_tokens)

# =====================================================
# STEP 11 : POS TAGGING
# =====================================================

print("\n" + "=" * 60)
print("STEP 11 : POS TAGGING")
print("=" * 60)

for i, sentence in enumerate(nltk_sentences, start=1):

    tokens = word_tokenize(sentence)

    tagged_tokens = pos_tag(tokens)

    print(f"\nSentence {i}")

    for word, tag in tagged_tokens:
        print(f"{word:15} --> {tag}")

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith("J"):
        return wordnet.ADJ

    elif treebank_tag.startswith("V"):
        return wordnet.VERB

    elif treebank_tag.startswith("N"):
        return wordnet.NOUN

    elif treebank_tag.startswith("R"):
        return wordnet.ADV

    else:
        return wordnet.NOUN
    
# =====================================================
# STEP 12 : LEMMATIZATION (WITH POS)
# =====================================================

print("\n" + "=" * 60)
print("STEP 12 : LEMMATIZATION (WITH POS)")
print("=" * 60)

for i, sentence in enumerate(nltk_sentences, start=1):

    tokens = word_tokenize(sentence)

    tagged_tokens = pos_tag(tokens)

    lemmas = []

    for word, tag in tagged_tokens:

        if word.isalpha():

            pos = get_wordnet_pos(tag)

            lemma = lemmatizer.lemmatize(word.lower(), pos)

            lemmas.append(lemma)

    print(f"\nSentence {i}")
    print(lemmas)


# =====================================================
# STEP 13 : NAMED ENTITY RECOGNITION (NER)
# =====================================================

print("\n" + "=" * 60)
print("STEP 13 : NAMED ENTITY RECOGNITION")
print("=" * 60)

for i, sentence in enumerate(nltk_sentences, start=1):

    print(f"\nSentence {i}")

    # Tokenize
    tokens = word_tokenize(sentence)

    # POS Tag
    tagged_tokens = pos_tag(tokens)

    # Named Entity Recognition
    entities = ne_chunk(tagged_tokens)

    print(entities)