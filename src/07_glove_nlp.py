from gensim.models import KeyedVectors

print("=" * 60)
print("LOAD GLOVE MODEL")
print("=" * 60)

glove_path = "models/glove.6B.100d.txt"

model = KeyedVectors.load_word2vec_format(
    glove_path,
    binary=False,
    no_header=True
)

print("Model Loaded.")

# =====================================================
# VOCABULARY SIZE
# =====================================================

print("\n" + "=" * 60)
print("VOCABULARY SIZE")
print("=" * 60)

print(len(model.key_to_index))

# =====================================================
# VECTOR SIZE
# =====================================================

print("\n" + "=" * 60)
print("VECTOR SIZE")
print("=" * 60)

print(model.vector_size)

# =====================================================
# WORD VECTOR
# =====================================================

print("\n" + "=" * 60)
print("WORD VECTOR")
print("=" * 60)

print(model["computer"])

# =====================================================
# MOST SIMILAR WORDS
# =====================================================

print("\n" + "=" * 60)
print("MOST SIMILAR WORDS")
print("=" * 60)

print(model.most_similar("computer", topn=5))

# =====================================================
# WORD SIMILARITY
# =====================================================

print("\n" + "=" * 60)
print("WORD SIMILARITY")
print("=" * 60)

print(
    model.similarity(
        "king",
        "queen"
    )
)

# =====================================================
# ANALOGY
# =====================================================

print("\n" + "=" * 60)
print("WORD ANALOGY")
print("=" * 60)

print(
    model.most_similar(
        positive=["king", "woman"],
        negative=["man"],
        topn=5
    )
)