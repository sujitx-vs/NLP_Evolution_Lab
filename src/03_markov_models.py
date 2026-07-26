import nltk
import re
from collections import defaultdict

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

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

print("\n" + "=" * 60)
print("SENTENCE TOKENIZATION")
print("=" * 60)

sentences = sent_tokenize(text)

for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence}")

# -----------------------------------------------------
# WORD TOKENIZATION
# -----------------------------------------------------

print("\n" + "=" * 60)
print("WORD TOKENIZATION")
print("=" * 60)

tokenized_sentences = []

for sentence in sentences:

    tokens = word_tokenize(sentence)

    tokenized_sentences.append(tokens)

print(tokenized_sentences)

# -----------------------------------------------------
# LOWERCASING
# -----------------------------------------------------

print("\n" + "=" * 60)
print("LOWERCASING")
print("=" * 60)

lower_document = []

for sentence in tokenized_sentences:

    lower_tokens = []

    for token in sentence:

        lower_tokens.append(token.lower())

    lower_document.append(lower_tokens)

print(lower_document)

# -----------------------------------------------------
# REMOVE PUNCTUATION
# -----------------------------------------------------

print("\n" + "=" * 60)
print("REMOVE PUNCTUATION")
print("=" * 60)

clean_document = []

pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*|\d+(?:\.\d+)?"

for sentence in lower_document:

    cleaned_tokens = []

    for token in sentence:

        if re.fullmatch(pattern, token):
            cleaned_tokens.append(token)

    clean_document.append(cleaned_tokens)

print(clean_document)

# -----------------------------------------------------
# BUILD BIGRAM TRANSITION COUNTS
# -----------------------------------------------------

print("\n" + "=" * 60)
print("BIGRAM TRANSITION COUNTS")
print("=" * 60)

transition_counts = defaultdict(lambda: defaultdict(int))

for sentence in clean_document:

    for i in range(len(sentence) - 1):

        current_word = sentence[i]
        next_word = sentence[i + 1]

        transition_counts[current_word][next_word] += 1

for word in transition_counts:

    print(f"\n{word}")

    for next_word, count in transition_counts[word].items():

        print(f"   -> {next_word:15} {count}")


# =====================================================
# TRANSITION PROBABILITIES
# =====================================================

print("\n" + "=" * 60)
print("TRANSITION PROBABILITIES")
print("=" * 60)

transition_probabilities = {}

for current_word, next_words in transition_counts.items():

    # Total outgoing transitions from the current word
    total = sum(next_words.values())

    transition_probabilities[current_word] = {}

    print(f"\nCurrent Word : {current_word}")
    print("-" * 40)

    for next_word, count in next_words.items():

        probability = count / total

        transition_probabilities[current_word][next_word] = probability

        print(f"{current_word:15} --> {next_word:15} = {probability:.4f}")


# =====================================================
# NEXT WORD PREDICTION
# =====================================================

print("\n" + "=" * 60)
print("NEXT WORD PREDICTION")
print("=" * 60)

while True:

    current_word = input("\nEnter a word (or type 'exit'): ").lower()

    if current_word == "exit":
        break

    if current_word not in transition_probabilities:
        print("Word not found in the corpus.")
        continue

    print(f"\nPossible next words after '{current_word}'\n")

    for next_word, probability in transition_probabilities[current_word].items():

        print(f"{next_word:20} {probability:.4f}")

    prediction = max(
        transition_probabilities[current_word],
        key=transition_probabilities[current_word].get
    )

    print(f"\nMost Probable Next Word : {prediction}")


# =====================================================
# SENTENCE GENERATION
# =====================================================

print("\n" + "=" * 60)
print("SENTENCE GENERATION")
print("=" * 60)

while True:

    start_word = input("\nEnter a starting word (or type 'exit'): ").lower()

    if start_word == "exit":
        break

    if start_word not in transition_probabilities:
        print("Word not found in the corpus.")
        continue

    sentence = [start_word]

    current_word = start_word

    max_words = 20

    for _ in range(max_words):

        next_words = transition_probabilities[current_word]

        # Select the word with highest probability
        next_word = max(
            next_words,
            key=next_words.get
        )

        sentence.append(next_word)

        current_word = next_word

        if current_word not in transition_probabilities:
            break

    print("\nGenerated Sentence:\n")
    print(" ".join(sentence))