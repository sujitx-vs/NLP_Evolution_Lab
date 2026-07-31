import re
import numpy as np
from gensim.models import KeyedVectors

# =====================================================
# LOAD GLOVE MODEL
# =====================================================

print("=" * 60)
print("LOADING GLOVE MODEL")
print("=" * 60)

glove_path = "models/glove.6B.100d.txt"

model = KeyedVectors.load_word2vec_format(
    glove_path,
    binary=False,
    no_header=True
)

print("Model Loaded Successfully.")

# =====================================================
# INPUT TEXT
# =====================================================

file_path = "datasets/txt_sample01.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("=" * 60)
print("ORIGINAL TEXT")
print("=" * 60)
print(text)


# =====================================================
# TOKENIZATION
# =====================================================

# Simple tokenizer (replace with your tokenizer later)
tokens = re.findall(r"\b[\w@.$'-]+\b", text.lower())

print("\nTokens:")
print(tokens)

# =====================================================
# CREATE EMBEDDINGS
# =====================================================

embedding_size = model.vector_size

embeddings = []

for token in tokens:

    if token in model:
        embeddings.append(model[token])

    else:
        print(f"OOV Word : {token}")
        embeddings.append(np.zeros(embedding_size))

embeddings = np.array(embeddings)

# =====================================================
# SAVE AS NUMPY FILE
# =====================================================

output_file = "dataset/processed/glove_embeddings.npy"

np.save(output_file, embeddings)

print("\nEmbeddings Saved Successfully.")

print("Shape :", embeddings.shape)
print("Saved To :", output_file)