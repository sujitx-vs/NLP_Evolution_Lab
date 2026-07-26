import nltk
import re
from collections import defaultdict

from nltk.tokenize import sent_tokenize, word_tokenize

# -----------------------------------------------------
# DOWNLOAD REQUIRED DATA
# -----------------------------------------------------

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

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

# -----------------------------------------------------
# POS TAGGING
# -----------------------------------------------------

print("\n" + "=" * 60)
print("POS TAGGING")
print("=" * 60)

tagged_sentences = []

for sentence in processed_sentences:

    tagged = nltk.pos_tag(sentence)

    tagged_sentences.append(tagged)

    print()

    for word, tag in tagged:
        print(f"{word:15} {tag}")

# =====================================================
# INITIAL STATE COUNTS
# =====================================================

print("\n" + "=" * 60)
print("INITIAL STATE COUNTS")
print("=" * 60)

initial_counts = defaultdict(int)

for sentence in tagged_sentences:

    first_tag = sentence[0][1]

    initial_counts[first_tag] += 1

for tag, count in initial_counts.items():
    print(f"{tag:10} {count}")

# =====================================================
# TRANSITION COUNTS
# =====================================================

print("\n" + "=" * 60)
print("TRANSITION COUNTS")
print("=" * 60)

transition_counts = defaultdict(lambda: defaultdict(int))

for sentence in tagged_sentences:

    tags = [tag for word, tag in sentence]

    for i in range(len(tags) - 1):

        current_tag = tags[i]
        next_tag = tags[i + 1]

        transition_counts[current_tag][next_tag] += 1

for current_tag in transition_counts:

    print(f"\n{current_tag}")

    for next_tag, count in transition_counts[current_tag].items():

        print(f"   -> {next_tag:10} {count}")

# =====================================================
# EMISSION COUNTS
# =====================================================

print("\n" + "=" * 60)
print("EMISSION COUNTS")
print("=" * 60)

emission_counts = defaultdict(lambda: defaultdict(int))

for sentence in tagged_sentences:

    for word, tag in sentence:

        emission_counts[tag][word] += 1

for tag in emission_counts:

    print(f"\n{tag}")

    for word, count in emission_counts[tag].items():

        print(f"   -> {word:15} {count}")

# =====================================================
# TRANSITION PROBABILITIES
# =====================================================

print("\n" + "=" * 60)
print("TRANSITION PROBABILITIES")
print("=" * 60)

transition_probability = {}

for current_tag, next_tags in transition_counts.items():

    transition_probability[current_tag] = {}

    total = sum(next_tags.values())

    print(f"\nCurrent Tag : {current_tag}")
    print("-" * 40)

    for next_tag, count in next_tags.items():

        probability = count / total

        transition_probability[current_tag][next_tag] = probability

        print(f"{current_tag:10} -> {next_tag:10} = {probability:.4f}")

# =====================================================
# EMISSION PROBABILITIES
# =====================================================

print("\n" + "=" * 60)
print("EMISSION PROBABILITIES")
print("=" * 60)

emission_probability = {}

for tag, words in emission_counts.items():

    emission_probability[tag] = {}

    total = sum(words.values())

    print(f"\nHidden State : {tag}")
    print("-" * 40)

    for word, count in words.items():

        probability = count / total

        emission_probability[tag][word] = probability

        print(f"{tag:10} emits {word:15} = {probability:.4f}")