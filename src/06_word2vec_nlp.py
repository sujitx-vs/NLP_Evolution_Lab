import nltk
import re

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec

# -----------------------------------------------------
# DOWNLOAD REQUIRED DATA
# -----------------------------------------------------

nltk.download("punkt")
nltk.download("punkt_tab")

# -----------------------------------------------------
# READ DATASET
# -----------------------------------------------------

file_path = "datasets/txt_sample01.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("=" * 60)
print("ORIGINAL TEXT")
print("=" * 60)
print(text)

# -----------------------------------------------------
# SENTENCE TOKENIZATION
# -----------------------------------------------------

sentences = sent_tokenize(text)

# -----------------------------------------------------
# WORD TOKENIZATION + CLEANING
# -----------------------------------------------------

pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*|\d+(?:\.\d+)?"

processed_sentences = []

for sentence in sentences:

    tokens = word_tokenize(sentence)

    cleaned_tokens = []

    for token in tokens:

        token = token.lower()

        if re.fullmatch(pattern, token):
            cleaned_tokens.append(token)

    if cleaned_tokens:
        processed_sentences.append(cleaned_tokens)

print("\n" + "=" * 60)
print("PROCESSED SENTENCES")
print("=" * 60)

for sentence in processed_sentences:
    print(sentence)

# -----------------------------------------------------
# TRAIN WORD2VEC
# -----------------------------------------------------

print("\n" + "=" * 60)
print("TRAIN WORD2VEC")
print("=" * 60)

model = Word2Vec(
    sentences=processed_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1          # 0 = CBOW, 1 = Skip-Gram
)

print("Training Completed.")

# -----------------------------------------------------
# VOCABULARY
# -----------------------------------------------------

print("\n" + "=" * 60)
print("VOCABULARY")
print("=" * 60)

print(model.wv.index_to_key)

# -----------------------------------------------------
# WORD VECTOR
# -----------------------------------------------------

print("\n" + "=" * 60)
print("WORD VECTOR")
print("=" * 60)

print("\nVector for 'nlp'\n")
print(model.wv["nlp"])

# -----------------------------------------------------
# VECTOR SIZE
# -----------------------------------------------------

print("\n" + "=" * 60)
print("VECTOR SIZE")
print("=" * 60)

print(len(model.wv["nlp"]))

# -----------------------------------------------------
# SIMILAR WORDS
# -----------------------------------------------------

print("\n" + "=" * 60)
print("MOST SIMILAR WORDS")
print("=" * 60)

print(model.wv.most_similar("nlp", topn=5))

# -----------------------------------------------------
# WORD SIMILARITY
# -----------------------------------------------------

print("\n" + "=" * 60)
print("WORD SIMILARITY")
print("=" * 60)

print(
    model.wv.similarity(
        "nlp",
        "interesting"
    )
)

# -----------------------------------------------------
# SAVE MODEL
# -----------------------------------------------------

model.save("models/word2vec.model")

print("\nModel Saved.")