import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv("datasets/rnn_lstm_next_word_dataset.csv")

print(df.head())
print("\nDataset size:", len(df))
print("Target classes:", df["target"].nunique())


# ==========================================================
# 2. INPUT AND TARGET
# ==========================================================

X = df["text"].astype(str)
y = df["target"].astype(str)


# ==========================================================
# 3. ENCODE TARGET WORDS
# ==========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)

print("\nPossible target words:")
print(label_encoder.classes_)


# ==========================================================
# 4. TOKENIZE INPUT TEXT
# ==========================================================

vocab_size = 5000

tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X)

X_sequences = tokenizer.texts_to_sequences(X)


# ==========================================================
# 5. PAD SEQUENCES
# ==========================================================

max_length = 40

X_padded = pad_sequences(
    X_sequences,
    maxlen=max_length,
    padding="pre",
    truncating="pre"
)


# ==========================================================
# 6. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_padded,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ==========================================================
# 7. BUILD VANILLA RNN
# ==========================================================

model = Sequential([

    Embedding(
        input_dim=vocab_size,
        output_dim=64,
        mask_zero=True
    ),

    SimpleRNN(
        64
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        num_classes,
        activation="softmax"
    )
])


# ==========================================================
# 8. COMPILE
# ==========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================================
# 9. TRAIN
# ==========================================================

history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=16,
    validation_split=0.20,
    verbose=1
)


# ==========================================================
# 10. EVALUATE
# ==========================================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nRNN Test Accuracy:", accuracy)


# ==========================================================
# 11. TEST LONG-DEPENDENCY SENTENCE
# ==========================================================

sentence = [
    "I grew up in France and after moving abroad for many years I still speak fluent"
]

sequence = tokenizer.texts_to_sequences(sentence)

padded = pad_sequences(
    sequence,
    maxlen=max_length,
    padding="pre",
    truncating="pre"
)

probabilities = model.predict(padded, verbose=0)

predicted_index = np.argmax(probabilities[0])

predicted_word = label_encoder.inverse_transform(
    [predicted_index]
)[0]

confidence = probabilities[0][predicted_index]


print("\nSentence:")
print(sentence[0])

print("\nRNN predicted next word:", predicted_word)
print("Confidence:", confidence)