import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense # type: ignore

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("datasets/sentiment_dataset.csv")

print(df.head())

# ==========================================================
# INPUT AND OUTPUT
# ==========================================================

X = df["text"].astype(str)

y = df["label"]

# ==========================================================
# ENCODE LABELS
# ==========================================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# ==========================================================
# TOKENIZATION
# ==========================================================

vocab_size = 10000

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

tokenizer.fit_on_texts(X)

X_sequences = tokenizer.texts_to_sequences(X)

# ==========================================================
# PADDING
# ==========================================================

max_length = 50

X_padded = pad_sequences(
    X_sequences,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_padded,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# BUILD RNN MODEL
# ==========================================================

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=100,
        input_length=max_length
    ),

    SimpleRNN(64),

    Dense(32, activation="relu"),

    Dense(1, activation="sigmoid")
])

# ==========================================================
# COMPILE
# ==========================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================================
# TRAIN
# ==========================================================

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# ==========================================================
# EVALUATE
# ==========================================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy :", accuracy)

# ==========================================================
# PREDICTION
# ==========================================================

sample = [
    "This movie was absolutely fantastic"
]

sample_seq = tokenizer.texts_to_sequences(sample)

sample_pad = pad_sequences(
    sample_seq,
    maxlen=max_length,
    padding="post"
)

prediction = model.predict(sample_pad)

print("\nPrediction Probability :", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Predicted Sentiment : Positive")
else:
    print("Predicted Sentiment : Negative")